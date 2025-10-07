import pandas as pd
import geopandas as gpd
import pybdshadow
import pytz
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pysolar.solar import get_altitude, get_azimuth
import matplotlib.colors as colors
from shapely import make_valid
from shapely.geometry import Polygon, box
from scipy.spatial import ConvexHull  # For masking to convex hull
import numpy as np
from scipy.interpolate import griddata  # For smooth interpolation
from map_extraction import compute_figsize
import scipy.ndimage as ndimage

def calculate_sunshine_minutes(file_path, lat1, lon1, lat2, lon2, date, hour, roofSelection, timestamp, show_plot=True, verbose=False):
    """
    Calculate and visualize sunshine minutes for a specific hour on a given date using the entire input area.
    
    Parameters:
    - file_path (str): Path to the GeoJSON file.
    - lat1 (float): Latitude of the first point (unused but kept for compatibility).
    - lon1 (float): Longitude of the first point (unused but kept for compatibility).
    - lat2 (float): Latitude of the second point (unused but kept for compatibility).
    - lon2 (float): Longitude of the second point (unused but kept for compatibility).
    - date (str): Date in 'YYYY-MM-DD' format (e.g., '2025-09-01').
    - hour (int): Hour of the day (0-23) for sunshine analysis.
    - roofSelection (bool): True for rooftop shadows (height > 0), False for ground shadows (height = 0).
    - timestamp (str): Timestamp for logging or output file naming (e.g., '20250901_1400').
    - show_plot (bool): Whether to show the plot (default True).
    - verbose (bool): If True, print per-minute details (default False to reduce output).
    
    Returns:
    - GeoDataFrame: Sunshine grid with sunshine minutes for the specified hour (or dummy for non-daylight).
    - fig, ax: Matplotlib figure and axis if show_plot=False.
    """
    # Read building data
    buildings = gpd.read_file(file_path)

    # The input building data must be a `GeoDataFrame` with the `height` column storing the building height information and the `geometry` column storing the geometry polygon information of building outline.

    # Preprocess the building data
    buildings = pybdshadow.bd_preprocess(buildings)

    # Use all buildings from the input file (no bounds filtering)
    buildings_analysis = buildings.copy()

    # Define strict bounds
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)

    # Step 4: Check if buildings_analysis is empty
    if buildings_analysis.empty:
        print(f"Warning: No buildings found in the input file! Timestamp: {timestamp}")
        # Create a default sunshine grid if no buildings
        sunshine = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([min_lon, min_lat], [min_lon, min_lat]),  # Arbitrary small grid within bounds
            crs="EPSG:4326"
        )
        sunshine['sunshine'] = 0
        sunshine['sunshine_minutes_in_hour'] = 0
    else:
        # Set up the grid structure for the entire area
        sunshine = pybdshadow.cal_sunshine(buildings_analysis,
                                          day=date,
                                          roof=roofSelection,
                                          accuracy=0.1,
                                          precision=1200)  # precision=1200s (20min) for daily calc
        
        # Clip sunshine grid to specified bounds to remove any extra points outside the lat-lon bounds
        bounds_box = box(min_lon, min_lat, max_lon, max_lat)
        sunshine = gpd.clip(sunshine, bounds_box)
        if sunshine.empty:
            print(f"Warning: Sunshine grid clipped to empty after bounds application for {roofSelection} shadows.")
            sunshine = gpd.GeoDataFrame(
                geometry=[], crs="EPSG:4326"
            )
            sunshine['sunshine'] = pd.Series(dtype=float)
            sunshine['sunshine_minutes_in_hour'] = pd.Series(dtype=float)

    # Step 5: Define the specific hour
    specific_time_start = datetime.strptime(f"{date} {hour}:00", '%Y-%m-%d %H:%M')
    specific_time_start = pytz.timezone('Asia/Hong_Kong').localize(specific_time_start)
    
    # Check solar altitude to determine if it's a daylight hour (for initial viz mode, but we always compute)
    centroid_lat = buildings_analysis.geometry.centroid.y.mean() if not buildings_analysis.empty else (lat1 + lat2) / 2
    centroid_lon = buildings_analysis.geometry.centroid.x.mean() if not buildings_analysis.empty else (lon1 + lon2) / 2
    altitude = get_altitude(centroid_lat, centroid_lon, specific_time_start.astimezone(pytz.UTC))

    # Step 6: Initialize a new column for sunshine minutes
    if 'sunshine_minutes_in_hour' not in sunshine.columns:
        sunshine['sunshine_minutes_in_hour'] = 0

    # Step 7: Always calculate per-minute sunshine (no skip based on hour start)
    times = [specific_time_start + timedelta(minutes=i) for i in range(60)]  # Every minute in the hour
    for t in times:
        # Check solar altitude for this specific minute
        utc_t = t.astimezone(pytz.UTC)
        minute_altitude = get_altitude(centroid_lat, centroid_lon, utc_t)
        if minute_altitude <= 0:
            # No sun: no sunshine for this minute, skip shadow calc
            is_sunlit = pd.Series([False] * len(sunshine), index=sunshine.index)
            if verbose:
                print(f"Time: {t}, No sun (altitude {minute_altitude:.2f}°), no sunshine")
        else:
            try:
                if not buildings_analysis.empty:
                    shadows = pybdshadow.bdshadow_sunlight(buildings_analysis, t, roof=roofSelection, include_building=False)
                else:
                    shadows = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                
                if shadows.empty:
                    is_sunlit = pd.Series([True] * len(sunshine), index=sunshine.index)
                else:
                    # Fix invalid geometries before union
                    shadows.geometry = shadows.geometry.apply(make_valid)
                    # Filter out invalid, empty, or degenerate (zero-area) geometries to prevent topology errors
                    shadows = shadows[
                        shadows.geometry.is_valid & 
                        (~shadows.geometry.is_empty) & 
                        (shadows.geometry.area > 1e-10)  # Tiny threshold for zero-area
                    ].copy()  # .copy() to avoid SettingWithCopyWarning
                    
                    if shadows.empty:
                        is_sunlit = pd.Series([True] * len(sunshine), index=sunshine.index)
                    else:
                        try:
                            shadow_union = shadows.union_all()
                            is_sunlit = ~sunshine.geometry.intersects(shadow_union)
                        except Exception as union_err:  # Catch any remaining union failures
                            print(f"Warning: Union failed ({union_err}), falling back to per-geometry intersects.")
                            # Fallback: Check intersects with each shadow individually (slower but robust)
                            is_sunlit = pd.Series([True] * len(sunshine), index=sunshine.index)
                            for _, shadow in shadows.iterrows():
                                is_sunlit &= ~sunshine.geometry.intersects(shadow.geometry)
                    
                    if verbose:
                        print(f"Time: {t}, Shadows calculated (altitude {minute_altitude:.2f}°)")
            except ValueError as e:
                if "Given time before sunrise or after sunset" in str(e):
                    # Fallback: treat as no sun
                    is_sunlit = pd.Series([False] * len(sunshine), index=sunshine.index)
                    if verbose:
                        print(f"Time: {t}, Library error (likely no sun), no sunshine")
                else:
                    raise e  # Re-raise if not the expected error
        
        # Add sunshine only if sunlit
        sunshine.loc[is_sunlit, 'sunshine_minutes_in_hour'] += 1

    # Step 8: Visualize based on computed sunshine (full map if any sunshine, else buildings-only)
    has_sunshine = sunshine['sunshine_minutes_in_hour'].max() > 0
    # Compute figsize based on bounds
    figsize = compute_figsize(min_lon, max_lon, min_lat, max_lat)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if has_sunshine:
        # Adjusted legend_kwds for larger colorbar (shrink=0.8 for slightly smaller, but to make scale "larger" in perception, use shrink=1.0 and adjust pad)
        sunshine.plot(
            ax=ax,
            column='sunshine_minutes_in_hour',
            cmap='plasma',
            alpha=1,
            norm=colors.Normalize(vmin=0, vmax=60),
            legend=True,
            legend_kwds={'label': "Sunshine Minutes", 'orientation': "vertical", 'shrink': 0.8, 'pad': 0.02}
        )
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax, edgecolor='k', facecolor=(0, 0, 0, 0))
    else:
        # Plot only building shapes if no sunshine in hour
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
        ax.set_title(f'Building Shapes - Hour {hour}:00 (No Sunlight), {date}, {roofSelection}')
    
    # Adjusted title with smaller font
    title_str = f'Sunshine Minutes in Hour ({hour}:00-{hour+1}:00, {date})' if has_sunshine else f'Building Shapes - Hour {hour}:00 (No Sunlight), {date}, {roofSelection}'
    ax.set_title(title_str, fontsize=12)
    
    # Set STRICT bounds (no buffer)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect('equal')

    # Adjust axis labels font size
    ax.set_xlabel('Longitude', fontsize=8)
    ax.set_ylabel('Latitude', fontsize=8)

    # Adjust tick font size
    ax.tick_params(axis='both', labelsize=9)

    # Adjust colorbar tick and label font sizes after plotting
    if has_sunshine:
        cbar = ax.get_legend()  # For geopandas plot legend
        if cbar:
            cbar.set_label('Sunshine Minutes', fontsize=10)
            for t in cbar.get_ticks():
                t.label.set_fontsize(10)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close without showing for compositing

    # Step 10: Print summary statistics (only if has sunshine)
    if has_sunshine:
        print(sunshine['sunshine_minutes_in_hour'].describe())
    
    return sunshine, fig, ax

def plot_combined_sunshine_overlay(ground_sunshine, roof_sunshine, buildings_analysis, hour, date, timestamp, base_path, min_lat, min_lon, max_lat, max_lon):
    """
    Plot combined sunshine visualization for ground and roof shadows overlaid in a single map.
    Uses rasterization via griddata for smooth overlay with transparency.
    Scale bar is 0 to 60 minutes for both layers.
    
    Parameters:
    - ground_sunshine: GeoDataFrame for ground shadow sunshine.
    - roof_sunshine: GeoDataFrame for roof shadow sunshine.
    - buildings_analysis: GeoDataFrame of buildings for overlay.
    - hour: The hour being processed.
    - date: The date string.
    - timestamp: Timestamp for file naming.
    - base_path: Base path for saving the figure.
    - min_lat, min_lon, max_lat, max_lon: Strict bounds for the map.
    
    Returns:
    - fig: Matplotlib figure.
    """
    if ground_sunshine.empty or roof_sunshine.empty:
        print("One of the sunshine GeoDataFrames is empty. Skipping overlay.")
        return None
    
    # NEW: Concatenate the two GeoDataFrames into a single one for unified visualization
    # Add a 'type' column to distinguish, but for visualization, we'll use the combined points
    # for a single interpolation layer. This avoids separate raster issues and NaN mismatches.
    ground_sunshine['type'] = 'ground'
    roof_sunshine['type'] = 'roof'
    combined_sunshine = pd.concat([ground_sunshine, roof_sunshine], ignore_index=True)
    # Explicitly set CRS after concatenation to avoid warnings
    combined_sunshine.crs = 'EPSG:4326'
    
    # Use strict bounds for grid (no padding)
    minx, miny, maxx, maxy = min_lon, min_lat, max_lon, max_lat
    
    # Higher grid resolution for smoother interpolation and fewer artifacts
    grid_res = 300  # Increased for finer detail
    
    x_coords = np.linspace(minx, maxx, grid_res)
    y_coords = np.linspace(miny, maxy, grid_res)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    # Prepare combined points using centroids
    combined_centroids = combined_sunshine.geometry.centroid
    points_combined = np.column_stack([combined_centroids.x.values, combined_centroids.y.values])
    values_combined = combined_sunshine['sunshine_minutes_in_hour'].values
    
    # Rasterize the combined data using griddata with 'nearest' method to avoid extrapolation to 0 at edges
    grid_combined = griddata(points_combined, values_combined, (grid_x, grid_y), method='nearest')
    
    # Clip to valid range [0, 60]
    grid_combined = np.clip(grid_combined, 0, 60)
    
    # NAN handling by using scipy.ndimage.distance_transform_edt
    valid_mask = ~np.isnan(grid_combined) # convert this NumPy array into binary np.array (1: fine, 0: NAN)
    if np.any(valid_mask):  # Only if there's at least one valid value
        dist, idx = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
        filled = grid_combined.copy()
        nan_mask = ~valid_mask # 1: NAN, 0: fine
        filled[nan_mask] = grid_combined[idx[0][nan_mask], idx[1][nan_mask]] # selects all the NaN (hole) positions in the filled array using nan_mask and 
                                                                             # replaces their values with data from the nearest valid positions in grid_combined, 
                                                                             # by indexing into it using the pre-computed row (idx[0]) and column (idx[1]) coordinates 
                                                                             # for those exact hole spots
        grid_combined = filled
    else:
        # All NaN: fill with 0 (no sunshine)
        grid_combined = np.zeros_like(grid_combined)

    # Compute figsize based on bounds
    figsize = compute_figsize(min_lon, max_lon, min_lat, max_lat)
    
    # Create figure with white background explicitly
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Define shared norm with clipping
    norm = colors.Normalize(vmin=0, vmax=60, clip=True)
    
    # Plot the combined grid as a single layer with full opacity (since concatenated, it blends naturally via overlapping points)
    im_combined = ax.pcolormesh(grid_x, grid_y, grid_combined,
                                cmap='plasma',
                                alpha=1.0,
                                norm=norm,  # Use explicit norm with clip
                                shading='auto')  # Changed to 'auto' to match dimensions
    
    # Overlay building outlines
    if not buildings_analysis.empty:
        buildings_analysis.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add a single colorbar for the shared scale, adjusted for size
    cbar = plt.colorbar(im_combined, ax=ax, shrink=0.8, pad=0.02, aspect=10)
    cbar.set_label('Sunshine Minutes (0-60)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Set STRICT limits (no margin, no extra space)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=9)
    ax.set_title(f'Combined Ground & Rooftop Sunshine\nHour {hour:02d}:00-{hour+1:02d}:00, {date}', fontsize=12)
    
    plt.tight_layout()
    
    # Save the combined overlay figure with white background
    combined_path = f"{base_path}_{timestamp}_combined_overlay_sunshine_hour_{hour:02d}.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Combined overlay sunshine map for hour {hour:02d} saved as: {combined_path}")
    
    plt.show()
    return fig