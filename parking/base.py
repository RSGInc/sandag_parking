import os
import pandas as pd
import geopandas as gpd
import yaml


class Base:

    reduced_parking_df = None
    imputed_parking_df = None
    districts_df = None
    estimated_spaces_df = None
    districts_dict = None
    combined_df = None

    # Columns in the "combined_df" at the end of the process
    step_cols = {}

    def __init__(self):
        with open("settings.yaml", "r") as stream:
            try:
                self.settings = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        assert self.settings.get("models")

        # Set defaults
        default_settings = {
            "space_estimation_method": "lm",
            "cache_dir": "./cache",
            "input_dir": "./data",
            "output_dir": "./output",
            "walk_dist": 0.5,
            "walk_coef": -0.3,
            "plot": True,
        }

        # Add default parameters if missing
        for key, value in default_settings.items():
            self.settings.setdefault(key, value)

        # Create cach directory if not already there
        if not os.path.exists(self.settings.get("cache_dir")):
            os.mkdir(self.settings.get("cache_dir"))

        if not os.path.exists(self.settings.get("plots_dir")):
            os.makedirs(self.settings.get("plots_dir"))

        # Initialize empty vars
        self.estimated_spaces = None
        self.full_graph = None
        self.street_data = None
        self.mgra_gdf = None

        # Input data
        inputs = self.settings.get('inputs')
        raw_path = inputs.get("raw_parking_inventory")
        lu_path = inputs.get("land_use")

        self.raw_parking_df = pd.read_csv(raw_path).set_index("mgra")
        self.lu_df = pd.read_csv(lu_path).set_index("mgra")
        self.update_combined_df("landuse_df", self.lu_df)

        self.read_existing_data()

    def mgra_data(self):
        if self.mgra_gdf is None:
            print("Reading MGRA shapefile data")
            path = self.settings.get("geometry")
            cached_path = os.path.join(
                self.settings.get("cache_dir"), "cached_mgra.shp"
            )

            if not os.path.isfile(cached_path):
                self.mgra_gdf = gpd.read_file(path).set_index("MGRA")[
                    ["TAZ", "geometry"]
                ]
                self.mgra_gdf.to_file(cached_path)
            else:
                self.mgra_gdf = gpd.read_file(cached_path).set_index("MGRA")

        return self.mgra_gdf

    def update_combined_df(self, model_step: str, df: pd.DataFrame, include_lu_cols: bool = False):
        """
        Update the combined dataframe with a new dataframe
        """
        if self.combined_df is None or self.combined_df.empty:
            self.combined_df = df
        else:
            self.combined_df = self.combined_df.join(df)

        # Save the columns for this step
        if include_lu_cols:
            self.step_cols[model_step] = self.combined_df.columns
        else:
            # Append the last step columns except for the land use columns
            prev_cols = []
            for k, v in self.step_cols.items():
                if k != "landuse_df":
                    prev_cols += v

            self.step_cols[model_step] = prev_cols + list(df.columns)

        return

    def read_existing_data(self):
        """
        Read existing data from the output directory if it exists
        """
        # Read existing data
        for df_name, path in self.settings.get('outputs').items():
            if os.path.exists(path):
                setattr(self, df_name, pd.read_csv(path))
        return

    def write_output(self):

        output_cols = self.settings.get('output_columns')

        for df_name, out_path in self.settings.get('outputs').items():

            # Check if the dataframe is in the combined dataframe
            assert isinstance(self.combined_df, pd.DataFrame)

            if df_name in self.step_cols.keys():
                df = self.combined_df[self.step_cols[df_name]]

            elif hasattr(self, df_name):
                df = getattr(self, df_name).reset_index()

            else:
                raise ValueError(f"Dataframe {df_name} not found")

            if df_name in output_cols.keys():
                # Format column names
                renaming = {k: k if v is None else v for k, v in output_cols[df_name].items()}
                df = df.rename(columns=renaming)[renaming.values()]
                df.fillna(0, inplace=True)

            df.to_csv(out_path, index=False)

        return
