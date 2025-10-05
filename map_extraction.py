import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from shapely.geometry import Polygon

def extract_map_subset(file_path, lat1, lon1, lat2, lon2, timestamp, csv_path):
    """
    Extract and visualize a subset of the GeoJSON map within the region defined by two points.
    Clips building geometries to the bounding box for buildings that intersect it.
    Merges heights from CSV, updates TOPHEIGHT and BASEHEIGHT, calculates height, and uses a normal linear color scale.
    Exports the subset GeoDataFrame to CSV, saves the plot as PNG, and saves the subset as a new GeoJSON file.
    
    Parameters:
    - file_path (str): Path to the GeoJSON file.
    - lat1 (float): Latitude of the first point.
    - lon1 (float): Longitude of the first point.
    - lat2 (float): Latitude of the second point.
    - lon2 (float): Longitude of the second point.
    - timestamp (str): Timestamp for output file naming.
    - csv_path (str): Path to the CSV file with heights.
    
    Returns:
    - GeoDataFrame: Subset of the original data within the bounding box with updated heights and clipped geometries.
    - str: Path to the extracted GeoJSON file.
    """
    # Step 1: Load the GeoJSON file, assuming EPSG:2326 (Hong Kong 1980 Grid)
    gdf = gpd.read_file(file_path, crs='EPSG:2326')
    
    # Step 2: Reproject to WGS84 (EPSG:4326) for consistency
    gdf = gdf.to_crs('EPSG:4326')
    
    # Step 3: Define the bounding box from the two points
    minx, maxx = min(lon1, lon2), max(lon1, lon2)
    miny, maxy = min(lat1, lat2), max(lat1, lat2)
    
    # Create a bounding box polygon for clipping
    bbox_polygon = gpd.GeoDataFrame(
        geometry=[Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)])],
        crs='EPSG:4326'
    )
    
    # Step 4: Clip the geometries to the bounding box (selects intersecting and clips them)
    subset_gdf = gpd.clip(gdf, bbox_polygon)
    
    # Step 5: Load heights from CSV and merge
    if os.path.exists(csv_path):
        heights_df = pd.read_csv(csv_path)
        # Ensure consistent OBJECTID type for merge
        subset_gdf['OBJECTID'] = subset_gdf['OBJECTID'].astype(str)
        heights_df['OBJECTID'] = heights_df['OBJECTID'].astype(str)
        # Convert columns to numeric, handling empty strings as NaN
        heights_df['TOPHEIGHT'] = pd.to_numeric(heights_df['TOPHEIGHT'], errors='coerce')
        heights_df['BASEHEIGHT'] = pd.to_numeric(heights_df['BASEHEIGHT'], errors='coerce')
        
        # Merge on OBJECTID, keeping all subset buildings
        subset_gdf = subset_gdf.merge(heights_df, on='OBJECTID', how='left', suffixes=('', '_csv'))
        
        # Update TOPHEIGHT and BASEHEIGHT with CSV values where available
        subset_gdf['TOPHEIGHT'] = subset_gdf.apply(
            lambda row: row['TOPHEIGHT_csv'] if pd.notnull(row['TOPHEIGHT_csv']) else row['TOPHEIGHT'], axis=1
        )
        subset_gdf['BASEHEIGHT'] = subset_gdf.apply(
            lambda row: row['BASEHEIGHT_csv'] if pd.notnull(row['BASEHEIGHT_csv']) else row['BASEHEIGHT'], axis=1
        )
        
        # Drop temporary merge columns
        subset_gdf = subset_gdf.drop(columns=['TOPHEIGHT_csv', 'BASEHEIGHT_csv'])
        
        print("Heights merged from CSV.")
    else:
        print(f"Warning: CSV file {csv_path} not found. Using existing heights.")
    
    # Step 6: Calculate height (TOPHEIGHT - BASEHEIGHT), treating null as 0
    subset_gdf['TOPHEIGHT'] = pd.to_numeric(subset_gdf['TOPHEIGHT'], errors='coerce').fillna(30) # purpose: testing the roofshadow
    subset_gdf['BASEHEIGHT'] = pd.to_numeric(subset_gdf['BASEHEIGHT'], errors='coerce').fillna(0)
    subset_gdf['height'] = subset_gdf['TOPHEIGHT'] - subset_gdf['BASEHEIGHT']
    
    # Step 7: Clip heights to 0 to max height in subset
    max_height = subset_gdf['height'].max() if not subset_gdf.empty else 500  # Default to 500 if empty
    subset_gdf['height'] = np.clip(subset_gdf['height'], 0, max_height)
    
    # Step 8: Export subset GeoDataFrame to CSV and GeoJSON, and save plot as PNG
    geojson_path = None
    if not subset_gdf.empty:
        # Define output file paths with timestamp
        base_path = os.path.splitext(file_path)[0]
        csv_path_out = f"{base_path}_{timestamp}_extracted_map_relheight.csv"
        geojson_path = f"{base_path}_{timestamp}_extracted_map_relheight.geojson"
        png_path = f"{base_path}_{timestamp}_extracted_map_relheight.png"
        
        # Export to CSV (include geometry as WKT)
        subset_gdf.loc[:, 'geometry_wkt'] = subset_gdf['geometry'].apply(lambda x: x.wkt if x is not None else '')
        subset_gdf.to_csv(csv_path_out, index=False)
        print(f"Subset GeoDataFrame exported to: {os.path.abspath(csv_path_out)}")
        
        # Export to GeoJSON with updated TOPHEIGHT and BASEHEIGHT
        subset_gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"Subset GeoDataFrame exported to: {os.path.abspath(geojson_path)}")
        
        # Visualize the subset with NORMAL LINEAR color scale
        fig, ax = plt.subplots(1, figsize=(12, 12))
        # Plot all buildings with colormap, including height=0 as the minimum color
        subset_gdf.plot(
            ax=ax,
            column='height',
            cmap='YlOrBr',
            alpha=0.8,
            norm=colors.Normalize(vmin=0, vmax=max_height),
            legend=True,
            legend_kwds={'label': "Building Height (m)", 'orientation': "vertical", 'shrink': 0.5}
        )
        # Overlay edges on top for clear shapes
        subset_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor='black',
            linewidth=0.5
        )
        
        # Center the plot without extra padding for a tightly fitted view
        if not subset_gdf.empty:
            bounds = subset_gdf.total_bounds
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            ax.set_xlim(center_x - (width / 2), center_x + (width / 2))
            ax.set_ylim(center_y - (height / 2), center_y + (height / 2))
        
        ax.set_title(f'Building Footprints in Region ({minx:.3f}, {miny:.3f}) to ({maxx:.3f}, {maxy:.3f})', fontsize=16)
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')

        plt.tight_layout()
        
        # Save plot as PNG
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {os.path.abspath(png_path)}")
        
        # Show plot
        plt.show()
    else:
        print("No buildings found in the specified region.")
        # For empty, create empty paths or handle
        base_path = os.path.splitext(file_path)[0]
        geojson_path = f"{base_path}_{timestamp}_extracted_map_relheight.geojson"
        subset_gdf.to_file(geojson_path, driver='GeoJSON')
    
    # Step 9: Print basic info
    print(f"Number of features extracted: {len(subset_gdf)}")
    if not subset_gdf.empty:
        print(f"Extracted map bounds: {subset_gdf.total_bounds}")
        print(f"Height statistics:\n{subset_gdf['height'].describe()}")
        print(subset_gdf[['TOPHEIGHT', 'BASEHEIGHT', 'height', 'geometry']].head())
    
    # Step 10: Return the subset and the geojson path
    return subset_gdf, geojson_path