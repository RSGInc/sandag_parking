import numpy as np
import pandas as pd
from . import base
import folium


class PreprocessData(base.Base):
    """
    Preprocessing step for combined land use + parking data.

    This step replaces 'run_reduction' when the input data already contains
    aggregated parking costs rather than detailed on-street/off-street
    parking inventories.

    It uses the ``column_mapping`` in settings.yaml to rename input columns
    to the internal names expected by downstream steps.

    Expected input columns (after column mapping):
        hourly  - on-street parking cost per hour
        daily   - structured (off-street) parking daily cost
        monthly - structured (off-street) parking monthly cost
        spaces  - structured (off-street) parking spaces only
                  (on-street spaces are unknown and will be estimated)
    """
    
    def map_input_costs(self, parking_df, cost_col, prefix="input"):    
        """Interactive Folium choropleth of input parking cost data."""
        if not self.settings.get("plots"):
            return
        
         # Read input
        mgra_gdf = self.mgra_data()

        # Reset index so the zone ID becomes a regular column
        gdf = mgra_gdf[["geometry"]].join(parking_df[[cost_col]]).reset_index()
        
        # Drop external stations
        gdf = gdf = gdf.dropna(subset=cost_col)
        zone_col = gdf.columns[0]

        centroid = gdf.geometry.union_all().centroid
        map_center = self.settings.get("map_center", [centroid.y, centroid.x])
        mapplot = folium.Map(
            location=map_center,
            tiles=self.settings.get("map_tiles", "cartodbpositron"),
            zoom_start=self.settings.get("map_zoom", 10),
        )
        folium.Choropleth(
            data=gdf,
            geo_data=gdf,
            columns=[zone_col, cost_col],
            key_on=f"feature.properties.{zone_col}",
            fill_color="YlGnBu",
            line_weight=0.1,
            line_opacity=0.5,
            legend_name=f"Input {cost_col} cost",
        ).add_to(mapplot)

        # Add hover tooltip showing zone name and cost value
        folium.GeoJson(
            data=gdf,
            style_function=lambda _: {"fillOpacity": 0, "weight": 0},
            highlight_function=lambda _: {"fillOpacity": 0.5, "weight": 2},
            tooltip=folium.GeoJsonTooltip(
                fields=[zone_col, cost_col],
                aliases=["Zone:", f"Input {cost_col.title()} Cost:"],
                localize=True,
            ),
        ).add_to(mapplot)

        mapplot.save(f"{self.settings.get('plots_dir')}/{prefix}_{cost_col}_cost.html")

    def run_preprocessing(self):
        """
        Extract parking cost data from the (already column-mapped) land use
        dataframe and produce a ``reduced_parking_df`` compatible with the
        imputation step.

        Key behaviour:
        - ``spaces`` is kept as NaN for zones that have parking costs but no
          reported structured spaces.  The space-estimation step will later
          fill these in from network characteristics.
        - ``paid_spaces`` is set > 0 for every zone that has *any* non-zero
          parking cost, even if structured spaces are 0/unknown.  This
          ensures those zones are included in district creation.
        """
        print("Preprocessing input data")

        cost_cols = ["hourly", "daily", "monthly"]

        # --- Extract parking columns from land use ---
        available_cost_cols = [c for c in cost_cols if c in self.lu_df.columns]
        parking_cols = available_cost_cols + (
            ["spaces"] if "spaces" in self.lu_df.columns else []
        )
        parking_df = self.lu_df[parking_cols].copy()

        # Ensure spaces is numeric.  Keep NaN for zones without reported
        # structured spaces so that the space-estimation step fills them in.
        if "spaces" in parking_df.columns:
            parking_df["spaces"] = pd.to_numeric(
                parking_df["spaces"], errors="coerce"
            )
            # Treat 0 structured spaces the same as unknown
            parking_df.loc[parking_df["spaces"] == 0, "spaces"] = np.nan
        else:
            parking_df["spaces"] = np.nan
            
        # --- Plot input costs ---
        for cost in available_cost_cols:
           self.map_input_costs(parking_df, cost, prefix="input")

        # --- Adjust costs and identify zones with paid/free parking ---
        # SKATS adjustment
        # For zones with free parking, retain $0 cost
        if self.settings.get("is_skats"):
            eps_float = np.finfo(float).eps

            # Zones with free limited on-street parking have PRKCST_HR=0 & PRKCST_DAILY=15
            zero_hourly = parking_df['hourly'] == 0
            fine_daily = parking_df['daily'] == 15
            parking_df.loc[zero_hourly & fine_daily, 'hourly'] = eps_float
            
            # Zone 7330 has known free parking
            parking_df.loc[7330, ['hourly', 'daily', 'monthly']] = eps_float
            
        # Replace 0 costs with NaN – 0 means "no data", not "free parking"
        for cost in available_cost_cols:
            parking_df[cost] = pd.to_numeric(
                parking_df[cost], errors="coerce"
            ).replace(0, np.nan)
            
        # --- Plot preprocessed costs ---
        for cost in available_cost_cols:
           self.map_input_costs(parking_df, cost, prefix="preprocessed")

        # --- Identify zone categories ---
        has_cost = parking_df[available_cost_cols].notna().any(axis=1)
        has_spaces = parking_df["spaces"].notna() & (parking_df["spaces"] > 0)

        # paid_spaces: used by district-creation to identify paid-parking
        # zones.  For zones that have on-street costs but no reported
        # structured spaces, set to 1 so they are still included.
        parking_df["paid_spaces"] = 0
        parking_df.loc[has_cost & has_spaces, "paid_spaces"] = (
            parking_df.loc[has_cost & has_spaces, "spaces"]
        )
        parking_df.loc[has_cost & ~has_spaces, "paid_spaces"] = 1

        # free_spaces: structured spaces in zones with no associated cost
        parking_df["free_spaces"] = 0
        parking_df.loc[~has_cost & has_spaces, "free_spaces"] = (
            parking_df.loc[~has_cost & has_spaces, "spaces"]
        )

        # Keep only zones that have cost data or known structured spaces
        parking_df = parking_df[has_cost | has_spaces]

        # Store as the "reduced" parking dataframe consumed by imputation
        self.reduced_parking_df = parking_df

        # Remove parking cost columns from lu_df and combined_df to avoid
        # duplication when joining reduced_parking_df back in.
        drop_cols = [c for c in cost_cols + ["spaces"] if c in self.lu_df.columns]
        self.lu_df = self.lu_df.drop(columns=drop_cols, errors="ignore")
        self.combined_df = self.combined_df.drop(columns=drop_cols, errors="ignore")

        self.update_combined_df("reduced_parking_df", self.reduced_parking_df)

        print(f"  Total zones in input:      {len(self.lu_df)}")
        print(f"  Zones with parking data:   {len(parking_df)}")
        print(f"  Zones with paid parking:   {has_cost.sum()}")
        print(f"  Zones with known spaces:   {has_spaces.sum()}")
        print(f"  Zones needing space est.:  {(has_cost & ~has_spaces).sum()}")
