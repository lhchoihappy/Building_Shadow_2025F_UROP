import geopandas as gpd
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from shapely.geometry import Polygon

def compute_figsize(min_lon, max_lon, min_lat, max_lat, base_width=12):
    """
    Compute figsize based on the map aspect ratio to avoid margins with aspect='equal'.
    
    Parameters:
    - min_lon, max_lon, min_lat, max_lat: Bounds.
    - base_width: Base width for figsize (default 12).
    
    Returns:
    - tuple: (width, height) for figsize.
    """
    delta_lon = max_lon - min_lon
    delta_lat = max_lat - min_lat
    map_aspect = delta_lon / delta_lat if delta_lat > 0 else 1
    if map_aspect > 1:
        height = base_width / map_aspect
        return (base_width, height)
    else:
        width = base_width * map_aspect
        return (width, base_width)

def extract_map_subset(file_path, lat1, lon1, lat2, lon2, timestamp, csv_path):
    """
    Extract and visualize a subset of the GeoJSON map within the region defined by two points.
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
    - GeoDataFrame: Subset of the original data within the bounding box with updated heights.
    - str: Path to the extracted GeoJSON file.
    """
    # Step 1: Load the GeoJSON file, assuming EPSG:2326 (Hong Kong 1980 Grid)
    gdf = gpd.read_file(file_path, crs='EPSG:2326')
    
    # Step 2: Reproject to WGS84 (EPSG:4326) for consistency
    gdf = gdf.to_crs('EPSG:4326')
    
    # Step 3: Define the bounding box from the two points
    min_lon, max_lon = min(lon1, lon2), max(lon1, lon2)
    min_lat, max_lat = min(lat1, lat2), max(lat1, lat2)
    
    # Step 4: Extract subset using .cx indexer (spatial slicing)
    subset_gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]
    
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
    subset_gdf['TOPHEIGHT'] = pd.to_numeric(subset_gdf['TOPHEIGHT'], errors='coerce').fillna(30)
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
        
        # Compute figsize based on bounds
        figsize = compute_figsize(min_lon, max_lon, min_lat, max_lat, base_width=18)
        
        # Visualize the subset with NORMAL LINEAR color scale
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        # Plot filled colors first
        subset_gdf.plot(
            ax=ax,
            column='height',  # Use clipped height directly
            cmap='YlOrBr',    # Colormap similar to the image (yellow low, brown high)
            alpha=0.8,        # Slight transparency
            norm=colors.Normalize(vmin=0, vmax=max_height), # Normal linear scale to max height
            legend=True,
            legend_kwds={'label': "Building Height (m)", 'orientation': "vertical", 'shrink': 1.0}
        )
        # Overlay edges on top for clear shapes
        subset_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor='black',
            linewidth=0.5
        )
        
        # Set plot properties with STRICT bounds (no buffer)
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_title(f'Building Footprints in Region ({min_lon:.3f}, {min_lat:.3f}) to ({max_lon:.3f}, {max_lat:.3f})', fontsize=16)
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