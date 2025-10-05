import os
from datetime import datetime
import geopandas as gpd
from map_extraction import extract_map_subset
from sunshine_minute import calculate_sunshine_minutes, plot_combined_sunshine_overlay
from solar_irradiance import calculate_hourly_solar_irradiance
from visualization import generate_irradiance_map
from sunrise_sunset import get_sunrise_sunset
import matplotlib.pyplot as plt
import pybdshadow
import numpy as np

if __name__ == "__main__":
    # File path and parameters
    file_path = r'D:\@UST\UROP\2025 Fall\HKBuildings.geojson'
    # csv_path = r'D:\@UST\UROP\2025 Fall\Height_dataset_KTown.csv'  # Adjust to the actual CSV path
    csv_path = r"D:\@UST\UROP\2025 Fall\ktown_heights_summary_v2.csv" # New Height Database
    
    # lat1, lon1 = 22.28367, 114.12569  # Kennedy point 1
    # lat2, lon2 = 22.27856, 114.13082  # Kennedy point 2

    lat1, lon1 = 22.28300, 114.12678  # Point 1 (much more narrow)
    lat2, lon2 = 22.28276, 114.12805  # Point 2 (much more narrow)
    
    date = '2025-08-20'
    
    # Generate timestamp at runtime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist. Please ensure the GeoJSON file is in the specified directory.")
    else:
        # Run extract_map_subset once
        print("Running extract_map_subset...")
        subset_gdf, geojson_path = extract_map_subset(file_path, lat1, lon1, lat2, lon2, timestamp, csv_path)
        
        # Compute centroid
        if not subset_gdf.empty:
            region_centroid = subset_gdf.unary_union.centroid
            centroid_lat = region_centroid.y
            centroid_lon = region_centroid.x
        else:
            centroid_lat, centroid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
        
        # Compute sunrise and sunset
        sunrise_time, sunset_time = get_sunrise_sunset(centroid_lat, centroid_lon, date)
        
        # Compute hourly irradiance data once
        print("\nComputing hourly solar irradiance data...")
        buildings, hourly_irradiance = calculate_hourly_solar_irradiance(geojson_path, date, timestamp, centroid_lat, centroid_lon)
        
        # Load buildings_analysis once for reuse
        buildings_analysis = gpd.read_file(geojson_path)
        buildings_analysis = pybdshadow.bd_preprocess(buildings_analysis)
        
        base_path = os.path.splitext(geojson_path)[0]
        
        # Loop over 24 hours
        hh = np.array([7,12,17])
        for hour in hh:

        # for hour in range(24):
            print(f"\n--- Processing Hour {hour:02d}:00 - {hour+1:02d}:00 ---")
            
            # Compute sunshine for ground (roof=False)
            print("Running calculate_sunshine_minutes for ground...")
            ground_sunshine, _, _ = calculate_sunshine_minutes(
                geojson_path, lat1, lon1, lat2, lon2, date, hour, False, timestamp, show_plot=False, verbose=False
            )
            
            # Compute sunshine for roof (roof=True)
            print("Running calculate_sunshine_minutes for roof...")
            roof_sunshine, _, _ = calculate_sunshine_minutes(
                geojson_path, lat1, lon1, lat2, lon2, date, hour, True, timestamp, show_plot=False, verbose=False
            )
            
            # Generate combined overlay sunshine plot
            print("Generating combined overlay sunshine map...")
            combined_fig = plot_combined_sunshine_overlay(ground_sunshine, roof_sunshine, buildings_analysis, hour, date, timestamp, base_path)
            if combined_fig:
                plt.close(combined_fig)
            
            # Save individual GeoJSONs for inspection (optional)
            ground_geojson_path = f"{base_path}_{timestamp}_sunshine_ground_hour_{hour:02d}.geojson"
            roof_geojson_path = f"{base_path}_{timestamp}_sunshine_roof_hour_{hour:02d}.geojson"
            ground_sunshine.to_file(ground_geojson_path, driver='GeoJSON')
            roof_sunshine.to_file(roof_geojson_path, driver='GeoJSON')
            print(f"Individual sunshine GeoJSONs saved for hour {hour:02d}")
            
            # Irradiance map for this hour
            print("Generating irradiance map...")
            irr_fig, irr_ax = generate_irradiance_map(buildings, hourly_irradiance, hour, timestamp, geojson_path)
            # Save irradiance here too
            irr_path = f"{base_path}_{timestamp}_irradiance_hour_{hour:02d}.png"
            plt.figure(irr_fig.number)
            plt.savefig(irr_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Irradiance map for hour {hour:02d} saved as: {irr_path}")
            plt.close(irr_fig)
            
            # Optionally show individual plots (comment out if too many)
            # plt.show()
        
        # Print summary
        print("\n=== SOLAR SUMMARY FOR DAY ===")
        daylight_hours = sum(1 for h in hourly_irradiance if hourly_irradiance[h]['irradiance'] > 0)
        print(f"Total daylight hours with irradiance > 0: {daylight_hours}")
        
        # Print sunrise and sunset after summary
        print("\n" + "="*50)
        print("SUNRISE AND SUNSET TIMES")
        print("="*50)
        print(f"Sunrise on {date}: {sunrise_time} HKT")
        print(f"Sunset on {date}: {sunset_time} HKT")
        print("="*50 + "\n")