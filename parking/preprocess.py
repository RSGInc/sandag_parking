import numpy as np
import pandas as pd
from . import base


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

        # Replace 0 costs with NaN – 0 means "no data", not "free parking"
        for cost in available_cost_cols:
            parking_df[cost] = pd.to_numeric(
                parking_df[cost], errors="coerce"
            ).replace(0, np.nan)

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
