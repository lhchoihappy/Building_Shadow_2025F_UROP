import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from map_extraction import compute_figsize
import pandas as pd
import os
import geopandas as gpd

def plot_temperature_map(ground_sunshine, roof_sunshine, buildings_analysis, hour, date, timestamp, base_path, min_lat, min_lon, max_lat, max_lon, min_temp, max_temp):
    """
    Plot temperature approximation map based on sunshine minutes for ground and roof overlaid in a single map.
    Uses linear mapping: 0 mins -> min_temp, 60 mins -> max_temp.
    Uses direct geopandas plotting for consistency with combined sunshine boundaries.
    
    Parameters:
    - ground_sunshine: GeoDataFrame for ground sunshine.
    - roof_sunshine: GeoDataFrame for roof sunshine.
    - buildings_analysis: GeoDataFrame of buildings for overlay.
    - hour: The hour being processed.
    - date: The date string.
    - timestamp: Timestamp for file naming.
    - base_path: Base path for saving the figure.
    - min_lat, min_lon, max_lat, max_lon: Strict bounds for the map.
    - min_temp: Minimum daily temperature (default 26.5°C from HK Park).
    - max_temp: Maximum daily temperature (default 30.7°C from HK Park).
    
    Returns:
    - fig: Matplotlib figure.
    """
    if ground_sunshine.empty or roof_sunshine.empty:
        print("One of the sunshine GeoDataFrames is empty. Skipping temperature map.")
        return None
    
    # Concatenate the two GeoDataFrames for unified visualization
    ground_sunshine['type'] = 'ground'
    roof_sunshine['type'] = 'roof'
    combined_sunshine = pd.concat([ground_sunshine, roof_sunshine], ignore_index=True)
    combined_sunshine.crs = 'EPSG:4326'
    
    # Compute temperature from sunshine minutes
    delta_temp = max_temp - min_temp
    combined_sunshine['estimated_temp'] = min_temp + (combined_sunshine['sunshine_minutes_in_hour'] / 60.0) * delta_temp
    
    # Compute figsize based on bounds
    figsize = compute_figsize(min_lon, max_lon, min_lat, max_lat)
    
    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Direct plot on combined GDF to preserve exact point coverage and white gaps, matching combined sunshine
    has_temp_data = (combined_sunshine['estimated_temp'].max() > min_temp) or (combined_sunshine['estimated_temp'].min() < max_temp)
    if has_temp_data and not combined_sunshine.empty:
        # Plot without automatic legend to allow custom colorbar
        combined_sunshine.plot(
            ax=ax,
            column='estimated_temp',
            cmap='coolwarm',  # Blue (cool) to red (warm) colormap
            alpha=1.0,
            norm=colors.Normalize(vmin=min_temp, vmax=max_temp, clip=True),
            legend=False  # Disable automatic legend
        )
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax, edgecolor='k', facecolor=(0, 0, 0, 0))
        
        # Manually add colorbar with custom ticks at min and max
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=colors.Normalize(vmin=min_temp, vmax=max_temp, clip=True))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02, aspect=10)
        cbar.set_label('Estimated Temperature (°C)', fontsize=10)
        cbar.set_ticks([min_temp, max_temp])
        cbar.ax.tick_params(labelsize=11, rotation=0)
    else:
        # No variation: just buildings
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
        ax.set_title(f'Building Shapes - Hour {hour}:00 (No Temperature Variation), {date}')
    
    # Set strict limits
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=9)
    ax.set_title(f'Estimated Temperature Map (Linear Approx. from Sunshine)\nHour {hour:02d}:00-{hour+1:02d}:00, {date}', fontsize=12)
    
    plt.tight_layout()
    
    # Save the temperature map
    temp_path = f"{base_path}_{timestamp}_temperature_approx_hour_{hour:02d}.png"
    plt.savefig(temp_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Temperature approximation map for hour {hour:02d} saved as: {temp_path}")
    
    return fig