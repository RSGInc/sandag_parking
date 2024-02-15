import os
import folium
from folium import plugins
import numpy as np
import pandas as pd
import geopandas as gpd
import alphashape
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import base


class CreateDistricts(base.Base):
    districts_df = pd.DataFrame()
    combined_df = pd.DataFrame()
    districts_dict = {}

    def create_districts(self):

        out_dir = self.settings.get("output_dir")

        mgra_gdf = self.mgra_data()
        print("Creating parking districts")
        self.districts_dict = self.parking_districts(
            self.imputed_parking_df, mgra_gdf, self.settings.get("walk_dist")
        )

        self.districts_df = self.districts_dict['districts'].drop(columns=['geometry'])

        prev_dir = os.path.join(out_dir, 'districts.csv')
        same = False
        if os.path.exists(prev_dir):
            prev_df = pd.read_csv(prev_dir).set_index(self.districts_df.index.name)
            if prev_df.equals(self.districts_df):
                same = True

        all_shp_files = all(
            [os.path.exists(f"{out_dir}/{geo}.shp") for geo, _ in self.districts_dict.items()]
            )

        # Skip this step if nothing to update
        if all_shp_files and same:
            print("Using existing district data")
            self.districts_df = pd.read_csv(os.path.join(out_dir, 'districts.csv'))
        else:
            # Read input
            plots_dir = self.settings.get("plots_dir")

            # Create cach directory if not already there
            for geo in self.districts_dict.keys():
                self.save_districts(geo)

            # Save district data to csv
            self.map_districts(self.districts_dict, mgra_gdf, plots_dir)
            self.map_districts_pngs(self.districts_dict, mgra_gdf, plots_dir)

        # append combined
        assert isinstance(self.districts_df, pd.DataFrame), "districts_df must be a dataframe"
        assert isinstance(self.combined_df, pd.DataFrame), "combined_df must be a dataframe"

        # self.districts_df.to_csv(os.path.join(out_dir, 'districts.csv'))

        # self.combined_df = self.combined_df.join(self.districts_df)
        self.update_combined_df("districts_df", self.districts_df)

    def parking_districts(self, imputed_df, mgra_gdf, max_dist):

        # Check that mgra is the index
        if imputed_df.index.name != "mgra" and "mgra" in imputed_df.columns:
            imputed_df.set_index("mgra", inplace=True)

        assert imputed_df.index.name == "mgra", "Index must be MGRA"

        # 1. Spatially cluster zones with paid parking
        paid_zones = mgra_gdf.loc[
            imputed_df[imputed_df.paid_spaces > 0].index, ["TAZ", "geometry"]
        ]

        # Calculate similarity matrix of ever zone to zone pair on their geometries not centroid
        print("Calculating similarity matrix")
        data_matrix = np.zeros([len(paid_zones)] * 2)
        for i, z in enumerate(tqdm(paid_zones.index)):
            data_matrix[i, :] = (
                paid_zones.geometry.distance(paid_zones.loc[z].geometry) / 5280
            )

        # Run clustering model -- kind of excessive but whatever
        print("Clustering zones")
        model = AgglomerativeClustering(
            metric="precomputed",
            compute_full_tree=True,
            linkage="single",
            distance_threshold=max_dist,
            n_clusters=None,
        ).fit(data_matrix)

        # Create cluster label dataframe to join from
        parking_clusters_labels = pd.DataFrame(
            model.labels_, columns=["cluster_id"], index=paid_zones.index
        )
        parking_clusters = paid_zones.join(parking_clusters_labels)

        # 2. Create a concave hull for each cluster & add buffer
        print("Creating concave hulls")

        def concave_hull(geo):
            alpha = 2.5 / (max_dist * 5280)
            flat_coords = [xy for geo in geo.exterior for xy in geo.coords]
            return alphashape.alphashape(flat_coords, alpha)

        hull_geoms = (
            parking_clusters.groupby("cluster_id")
            .geometry.apply(concave_hull)
            .set_crs(mgra_gdf.crs.to_epsg())
            .to_frame("geometry")
        )
        hull_geoms.index.name = "hull_id"
        buffer_geoms = hull_geoms.geometry.buffer(max_dist * 5280).to_frame("geometry")

        # Consolidate overlapping geometries
        parents = {}
        for i, geom in enumerate(buffer_geoms.geometry):
            connected = geom.intersects(buffer_geoms.geometry)
            edges = buffer_geoms.index[connected].to_list()
            for x in edges:
                if x not in parents:
                    parents[x] = i

        district_ids = pd.DataFrame.from_dict(
            parents, orient="index", columns=["district_id"]
        )
        buffer_geoms = buffer_geoms.join(district_ids).dissolve("district_id")

        # 3. Spatial Join all zones within hulls
        print("Performing spatial joins")
        # Add cluster id
        parking_districts = mgra_gdf[["geometry"]].join(
            parking_clusters[["cluster_id"]]
        )

        # Add hull id
        parking_districts = parking_districts.sjoin(
            hull_geoms.geometry.reset_index(), how="left", predicate="within"
        ).drop(columns="index_right")

        # Add district
        parking_districts = parking_districts.sjoin(
            buffer_geoms.geometry.reset_index(), how="left", predicate="within"
        ).drop(columns="index_right")

        # Determine parking_zone_type
        # is_prkdistrict    = zone within parking district
        # is_noprkspace     = zone within parking district but has no parking spaces

        # filters
        is_district = ~parking_districts.district_id.isnull()
        is_nodata = ~parking_districts.index.isin(paid_zones.index)
        is_hull = ~parking_districts.hull_id.isnull()

        # Assign
        parking_districts["is_prkdistrict"] = False
        parking_districts["is_noprkspace"] = False
        parking_districts["is_buffer"] = False
        parking_districts['is_hull'] = False
        parking_districts.loc[is_district, "is_prkdistrict"] = True
        parking_districts.loc[is_hull & is_nodata, "is_noprkspace"] = True
        parking_districts.loc[is_district & ~is_hull, "is_buffer"] = True
        parking_districts.loc[is_hull, "is_hull"] = True

        # parking_type:
        # 1: parking constrained area: has cluster_id AND district_id
        # 2: buffer around parking constrained area which is used to include free spaces to average
        #   into parking cost calculation: has district_id but no cluster_id
        # 3: no parking cost: Has neither cluster_id nor district_id

        parking_districts['parking_type'] = None
        parking_districts.loc[
            ~parking_districts.cluster_id.isnull() & ~parking_districts.district_id.isnull(),
            "parking_type"] = 1
        parking_districts.loc[
            parking_districts.cluster_id.isnull() & ~parking_districts.district_id.isnull(),
            "parking_type"] = 2
        parking_districts['parking_type'] = parking_districts['parking_type'].fillna(3)

        output = {
            "districts": parking_districts,
            "hulls": hull_geoms,
            "buffered_hulls": buffer_geoms,
            "clusters": parking_clusters,
        }

        return output

    def map_districts(self, district_dict, mgra_gdf, plots_dir):
        parking_districts = district_dict["districts"]
        parking_hulls = district_dict["hulls"]
        parking_buffered_hulls = district_dict["buffered_hulls"]
        parking_clusters = district_dict["clusters"]

        print("Plotting parking districts to html")

        # Plot paid parking zones
        print(f"Saving parking zone type map to {plots_dir}/parking_zone_types.html")
        concave_hull_layer = folium.GeoJson(
            data=parking_districts[
                ~parking_districts.hull_id.isnull() & parking_districts.cluster_id.isnull()
                ].geometry,
            style_function=lambda x: {
                "fillColor": "#000000",
                "color": "#000000",
                "weight": 0.1,
                "fillOpacity": 0.75,
            },
            name="Concave Hull (with cost data)"
        )

        no_data_layer = folium.GeoJson(
            data=parking_districts[
                ~parking_districts.hull_id.isnull() & ~parking_districts.cluster_id.isnull()
                ].geometry,
            style_function=lambda x: {
                "fillColor": "#e41a1c",
                "color": "#e41a1c",
                "weight": 0.1,
                "fillOpacity": 0.5,
                "name": "No data",
            },
            name="Concave Hull (no cost data)"
        )

        buffer_layer = folium.GeoJson(
            data=parking_districts[
                parking_districts.hull_id.isnull() & ~parking_districts.district_id.isnull()
                ].geometry,
            style_function=lambda x: {
                "fillColor": "#000000",
                "color": "#000000",
                "weight": 0.1,
                "fillOpacity": 0.25,
            },
            name="Concave Hull Buffer"
        )

        # Add legend to the top right, make it fully expanded
        # Label each layer with the name of the parking zone type
        control_layer = folium.LayerControl(
            position="topright",
            collapsed=False,
            overlay=True,
            control=True,
            show=True,
            autoZIndex=True,
            sortLayers=True,
            name="Parking Zone Types",
            )

        # Create a custom legend for each layer
        legend_html = '''
            <div style="position: fixed; top: 150px; right: 10px; width: 225px; height: 70px;
                        border:2px solid grey; z-index:9999; font-size:14px;
                        background-color:white; opacity:0.8">
                &nbsp; <i class="fa fa-square" style="color:#1a1a1a" op></i> Concave Hull (no cost data) &nbsp; <br>
                &nbsp; <i class="fa fa-square" style="color:#ff8080"></i> Concave Hull (with cost data) &nbsp; <br>
                &nbsp; <i class="fa fa-square" style="color:#bfbfbf"></i> Concave Hull Buffer &nbsp; <br>
            </div>
        '''

        # Create a dictionary to store the map layers
        map_layers = {
            "Concave Hull (with cost data)": concave_hull_layer,
            "Concave Hull (no cost data)": no_data_layer,
            "Concave Hull Buffer": buffer_layer,
            "Parking Zone Types": control_layer,
            "MiniMap": plugins.MiniMap(),
            "Legend": folium.Element(legend_html)
        }

        types_map = folium.Map(
            location=[32.7521765494396, -117.11514883606573],
            tiles='cartodbpositron',
            zoom_start=9,
        )

        # Add the GeoJson layers to the map
        for layer_name, layer in map_layers.items():
            if layer_name == "Legend":
                types_map.get_root().html.add_child(layer)  # type: ignore
            else:
                types_map.add_child(layer)

        # Save map
        types_map.save(f"{plots_dir}/parking_zone_types.html")

        # Plot paid parking zones ------------------------------------------------------------
        print(f"Saving paid zones cluster map to {plots_dir}/1_paid_zones.html")
        # Incremental process map
        mapplot = folium.Map(
            location=[32.7521765494396, -117.11514883606573],
            tiles='cartodbpositron',
            # attr=attribution,
            zoom_start=9,
        )
        # Folium chlorepleth map of parking clusters with all clusters colored red
        folium.Choropleth(
            data=parking_clusters.reset_index(),
            geo_data=parking_clusters.reset_index(),  # data
            columns=["mgra", "cluster_id"],  # [key, value]
            key_on="feature.properties.mgra",
            fill_column="cluster_id",
            fill_color="Set3",  # cmap
            line_weight=0.1,  # line wight (of the border)      # type: ignore
            line_opacity=0.5,  # line opacity (of the border)   # type: ignore
            legend_name="Parking clusters",
        ).add_to(mapplot)
        # Plot the paid parking zones
        # folium.GeoJson(
        #     data=parking_clusters[parking_clusters.cluster_id >= 0].geometry,
        #     style_function=lambda x: {
        #         "fillColor": "#e41a1c",
        #         "color": "#e41a1c",
        #         "weight": 0.1,
        #         "fillOpacity": 0.9
        #     },
        # ).add_to(mapplot)
        # Save map
        mapplot.save(f"{plots_dir}/1_paid_zones.html")

        # Plot concave hull areas & the zones within them
        print(f"Saving concave hull map to {plots_dir}/2_concave_hull.html")

        folium.GeoJson(
            data=parking_hulls.geometry,
            style_function=lambda x: {
                "fillColor": "#000000",
                "color": "#000000",
                "weight": 0.5,
                "fillOpacity": 0.5,
            },
        ).add_to(mapplot)
        mapplot.save(f"{plots_dir}/2_concave_hull.html")

        # Plot buffered concave hulls
        print(f"Saving buffered hull map to {plots_dir}/3_buffered_hull.html")
        buffered_hulls = gpd.GeoSeries(
            parking_buffered_hulls.geometry.unary_union
        ).set_crs(mgra_gdf.crs.to_epsg())
        folium.GeoJson(
            data=buffered_hulls,
            style_function=lambda x: {
                "fillColor": "#000000",
                "color": "#000000",
                "weight": 0.5,
                "fillOpacity": 0.25,
            },
        ).add_to(mapplot)

        # Save map
        mapplot.save(f"{plots_dir}/3_buffered_hull.html")

        # Plot parking district zones
        print(f"Saving parking district map to {plots_dir}/4_parking_district.html")
        gdf_districts = parking_districts[
            ~parking_districts.district_id.isnull()
        ].geometry.unary_union
        gdf_districts = gpd.GeoSeries(gdf_districts).set_crs(mgra_gdf.crs.to_epsg())
        folium.GeoJson(
            data=gdf_districts,
            style_function=lambda x: {
                "fillColor": "#000000",
                "weight": 0,
                "fillOpacity": 0.25,
                },
        ).add_to(mapplot)

        # Add legend in the top right
        legend = plugins.MiniMap()
        mapplot.add_child(legend)

        mapplot.save(f"{plots_dir}/4_parking_district.html")

    def map_districts_pngs(self, district_dict, mgra_gdf, plots_dir):
        parking_districts = district_dict["districts"]
        parking_hulls = district_dict["hulls"]
        parking_buffered_hulls = district_dict["buffered_hulls"]
        parking_clusters = district_dict["clusters"]

        print("Plotting parking districts to PNGs")

        print(f"Saving parking district map to {plots_dir}/4_parking_district.png")
        fig, axes = plt.subplots(2, 2, figsize=(8, 7))
        for axrow in axes:
            for ax in axrow:
                ax.axis('off')
                mgra_gdf.geometry.plot(color='white', edgecolor='k', linewidth=0.125, ax=ax)
                ax.set_xlim(np.array([6.26, 6.31]) * 1e6)
                ax.set_ylim(np.array([1.82, 1.86])*1e6)
        parking_clusters.plot(
            column='cluster_id', alpha=0.5, ax=axes[0][0], legend=False
            ).set_title('Parking zone clusters')
        parking_hulls.geometry.reset_index().plot(
            column='hull_id', alpha=0.5, ax=axes[0][1], legend=False
            ).set_title('Concave hulls')
        parking_buffered_hulls.geometry.reset_index().plot(
            column='district_id', alpha=0.5, ax=axes[1][0], legend=False
            ).set_title('Buffered hulls')
        parking_districts[~parking_districts.district_id.isnull()].plot(
            column='district_id', alpha=0.5, ax=axes[1][1], legend=False
            ).set_title('Parking districts')

        fig.savefig(f'{plots_dir}/clustermethod.png', dpi=800)

    def save_districts(self, geo):
        print(f"Saving {geo} shapefile")
        # Create cach directory if not already there
        shp_dir = os.path.join(self.settings.get("output_dir"), "shapefiles")
        if not os.path.exists(shp_dir):
            os.mkdir(shp_dir)

        out_path = os.path.join(shp_dir, geo + ".shp")
        self.districts_dict[geo].to_file(out_path)
