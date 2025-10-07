import geopandas as gpd
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pytz
from pysolar.solar import get_altitude, get_azimuth
from pysolar import radiation
from datetime import datetime, timedelta
from map_extraction import compute_figsize

def get_direct_irradiance(lat, lon, t):
    """
    Calculate clear-sky direct horizontal irradiance (W/m²) using pysolar.
    
    Parameters:
    - lat (float): Latitude in degrees.
    - lon (float): Longitude in degrees.
    - t (datetime): Localized datetime object.
    
    Returns:
    - float: Direct horizontal irradiance in W/m².
    """
    utc_t = t.astimezone(pytz.UTC)
    altitude = get_altitude(lat, lon, utc_t)
    if altitude <= 0:
        return 0.0
    irradiance = radiation.get_radiation_direct(utc_t, altitude)
    return irradiance

def generate_irradiance_map(buildings, hourly_irradiance, specified_hour, timestamp, file_path, min_lat, min_lon, max_lat, max_lon):
    """
    Generate a single irradiance map for the specified hour with a uniform scale bar.
    
    Parameters:
    - buildings: GeoDataFrame of buildings.
    - hourly_irradiance: Dict of hourly irradiance data.
    - specified_hour: The hour for visualization.
    - timestamp: For file naming.
    - file_path: Path to the GeoJSON file for base path.
    - min_lat, min_lon, max_lat, max_lon: Strict bounds for the map.
    
    Returns:
    - fig, ax: Matplotlib figure and axis.
    """
    # Get the specified hour data
    specified_irradiance_data = hourly_irradiance[specified_hour]
    
    print(f"Generating irradiance map for hour {specified_hour:02d}: "
          f"{specified_irradiance_data['irradiance']:.1f} W/m²")
    
    # Use strict bounds for grid
    minx, miny, maxx, maxy = min_lon, min_lat, max_lon, max_lat
    
    # Adaptive grid density: finer for smaller regions to avoid artifacts
    dx = maxx - minx
    dy = maxy - miny
    grid_res = max(200, int(min(dx, dy) / 0.000005))  # ~200+ points, finer step
    x_coords = np.linspace(minx, maxx, grid_res)
    y_coords = np.linspace(miny, maxy, grid_res)
    
    # Create solar irradiance grid for the specified hour
    dt_spec = specified_irradiance_data['datetime'].astimezone(pytz.utc)
    
    # Create initial coarse points for calculation (to speed up)
    coarse_res = 50  # Balance computation
    coarse_x = np.linspace(minx, maxx, coarse_res)
    coarse_y = np.linspace(miny, maxy, coarse_res)
    coarse_xx, coarse_yy = np.meshgrid(coarse_x, coarse_y)
    coarse_points = np.column_stack([coarse_xx.ravel(), coarse_yy.ravel()])
    
    # Calculate irradiance on coarse points
    coarse_irradiance = []
    for point in coarse_points:
        lon, lat = point
        altitude = get_altitude(lat, lon, dt_spec)
        if altitude > 0:
            irradiance = radiation.get_radiation_direct(dt_spec, altitude)
        else:
            irradiance = 0
        coarse_irradiance.append(irradiance)
    coarse_irradiance = np.array(coarse_irradiance)
    
    # Interpolate to fine grid for smooth overlay
    fine_xx, fine_yy = np.meshgrid(x_coords, y_coords)
    fine_points = np.column_stack([fine_xx.ravel(), fine_yy.ravel()])
    points = np.column_stack([coarse_points[:, 0], coarse_points[:, 1]])  # lon, lat
    irradiance_fine = griddata(points, coarse_irradiance, fine_points, method='cubic', fill_value=0)
    irradiance_grid = irradiance_fine.reshape(len(y_coords), len(x_coords))
    
    # Determine global maximum irradiance for uniform scale
    global_max_irrad = max(data['irradiance'] for data in hourly_irradiance.values())
    
    # Compute figsize based on bounds
    figsize = compute_figsize(min_lon, max_lon, min_lat, max_lat)
    
    # Visualization - Single map with better transparency and uniform scale
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create smooth irradiance overlay first (full coverage)
    im = ax.pcolormesh(fine_xx, fine_yy, irradiance_grid, 
                      cmap='YlOrRd',  # Yellow-Orange-Red colormap
                      alpha=1.0,  # Full opacity for background
                      shading='auto',  # Smooth without artifacts
                      vmin=0, 
                      vmax=global_max_irrad)
    
    # Plot buildings as outlines only (no fill to avoid gray overriding the scale)
    buildings.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2, alpha=1.0)
    
    # Add colorbar for irradiance with uniform scale
    cbar = plt.colorbar(im, ax=ax, shrink=1.0, aspect=15)
    cbar.set_label('Solar Irradiance (W/m²)', fontsize=10)
    
    ax.set_title(f'Solar Irradiance Map\n{datetime.now().strftime("%Y-%m-%d")} {specified_hour:02d}:00 - {specified_irradiance_data["irradiance"]:.1f} W/m²', 
                 fontsize=12, pad=20)
    ax.set_xlabel('Longitude', fontsize=8)
    ax.set_ylabel('Latitude', fontsize=8)
    
    # Set STRICT bounds (no buffer)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect('equal')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    base_path = os.path.splitext(file_path)[0]
    irradiance_map_path = f"{base_path}_{timestamp}_irradiance_hour_{specified_hour:02d}.png"
    plt.savefig(irradiance_map_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Irradiance map for hour {specified_hour:02d} saved as: {irradiance_map_path}")
    
    return fig, ax

def calculate_hourly_solar_irradiance(file_path, date, timestamp, centroid_lat, centroid_lon):
    """
    Calculate hourly solar irradiance for the centroid of the extracted region.
    Computes data for all hours but does not visualize (use generate_irradiance_map for that).
    
    Parameters:
    - file_path (str): Path to the extracted GeoJSON file.
    - date (str): Date in 'YYYY-MM-DD' format for solar calculation.
    - timestamp (str): Timestamp for output file naming.
    - centroid_lat, centroid_lon: Pre-computed region centroid coordinates.
    
    Returns:
    - GeoDataFrame: The original building data.
    - dict: Hourly solar irradiance data for the centroid point.
    """
    # Step 1: Load the extracted building data
    buildings = gpd.read_file(file_path)
    
    print(f"Region centroid: Latitude {centroid_lat:.6f}, Longitude {centroid_lon:.6f}")
    
    # Step 3: Calculate hourly solar irradiance for the centroid
    hourly_irradiance = {}
    
    # Hong Kong timezone
    tz = pytz.timezone('Asia/Hong_Kong')
    
    for hour in range(24):
        # Create datetime object for each hour
        dt = datetime.strptime(f"{date} {hour:02d}:00:00", '%Y-%m-%d %H:%M:%S')
        dt = tz.localize(dt)
        
        # Convert to UTC for pysolar (required by the library)
        dt_utc = dt.astimezone(pytz.utc)
        
        # Calculate solar altitude and azimuth
        altitude = get_altitude(centroid_lat, centroid_lon, dt_utc)
        
        # Calculate direct solar irradiance (W/m²) using the correct function
        if altitude > 0:  # Sun above horizon
            irradiance = radiation.get_radiation_direct(dt_utc, altitude)
        else:
            irradiance = 0
        
        hourly_irradiance[hour] = {
            'datetime': dt,
            'altitude': altitude,
            'irradiance': irradiance
        }
        
        print(f"Hour {hour:02d}: {altitude:.1f}° altitude, {irradiance:.1f} W/m²")
    
    # Export hourly irradiance data to CSV
    irradiance_df = pd.DataFrame([
        {
            'hour': hour,
            'datetime': data['datetime'],
            'solar_altitude': data['altitude'],
            'solar_irradiance_wm2': data['irradiance']
        }
        for hour, data in hourly_irradiance.items()
    ])
    
    base_path = os.path.splitext(file_path)[0]
    irradiance_csv_path = f"{base_path}_{timestamp}_hourly_irradiance.csv"
    irradiance_df.to_csv(irradiance_csv_path, index=False)
    print(f"Hourly irradiance data exported to: {irradiance_csv_path}")
    
    return buildings, hourly_irradiance

