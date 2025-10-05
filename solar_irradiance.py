import geopandas as gpd
import pandas as pd
import pytz
from datetime import datetime
from pysolar.solar import get_altitude
from pysolar import radiation
import os

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