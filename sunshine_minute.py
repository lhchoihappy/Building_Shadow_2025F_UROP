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
import os

def calculate_sunshine_minutes(file_path, lat1, lon1, lat2, lon2, date, hour, roof, timestamp, show_plot=True, verbose=False):
    """
    Calculate and visualize sunshine minutes for a specific hour on a given date using the entire input area.
    Computes both ground and rooftop shadows, visualizes individually and combined.
    
    Parameters:
    - file_path (str): Path to the GeoJSON file.
    - lat1 (float): Latitude of the first point (unused but kept for compatibility).
    - lon1 (float): Longitude of the first point (unused but kept for compatibility).
    - lat2 (float): Latitude of the second point (unused but kept for compatibility).
    - lon2 (float): Longitude of the second point (unused but kept for compatibility).
    - date (str): Date in 'YYYY-MM-DD' format (e.g., '2025-09-01').
    - hour (int): Hour of the day (0-23) for sunshine analysis.
    - roof (bool): Whether to calculate for rooftop (True) or ground (False).
    - timestamp (str): Timestamp for logging or output file naming (e.g., '20250901_1400').
    - show_plot (bool): Whether to show the plot (default True).
    - verbose (bool): If True, print per-minute details (default False to reduce output).
    
    Returns:
    - ground_sunshine: GeoDataFrame for ground shadows.
    - roof_sunshine: GeoDataFrame for rooftop shadows.
    - figures_dict: Dictionary containing all three figures {'ground': fig_ground, 'roof': fig_roof, 'combined': combined_fig}
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
        ground_sunshine = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([min_lon, min_lat], [min_lon, min_lat]),  # Arbitrary small grid within bounds
            crs="EPSG:4326"
        )
        ground_sunshine['sunshine'] = 0
        ground_sunshine['sunshine_minutes_in_hour'] = 0

        roof_sunshine = ground_sunshine.copy()
    else:
        # Set up the grid structure for the entire area
        ground_sunshine = pybdshadow.cal_sunshine(buildings_analysis,
                                          day=date,
                                          roof=False,
                                          accuracy=0.1,
                                          precision=1200)  # precision=1200s (20min) for daily calc
        
        # Clip sunshine grid to specified bounds to remove any extra points outside the lat-lon bounds
        bounds_box = box(min_lon, min_lat, max_lon, max_lat)
        ground_sunshine = gpd.clip(ground_sunshine, bounds_box)
        if ground_sunshine.empty:
            print(f"Warning: Sunshine grid clipped to empty after bounds application for ground shadows.")
            ground_sunshine = gpd.GeoDataFrame(
                geometry=[], crs="EPSG:4326"
            )
            ground_sunshine['sunshine'] = pd.Series(dtype=float)
            ground_sunshine['sunshine_minutes_in_hour'] = pd.Series(dtype=float)

        # Ensure rooftop grid has sufficient coverage
        roof_sunshine = pybdshadow.cal_sunshine(buildings_analysis,
                                            day=date,
                                            roof=True,
                                            accuracy=0.1,
                                            precision=1200)

        # Make sure we clip to the actual building areas, not just the bounds
        building_union = buildings_analysis.unary_union
        if not building_union.is_empty:
            # Keep points that are within or near buildings
            roof_sunshine = roof_sunshine[roof_sunshine.geometry.intersects(building_union.buffer(0.0001))]
        
        # Clip sunshine grid to specified bounds to remove any extra points outside the lat-lon bounds
        bounds_box = box(min_lon, min_lat, max_lon, max_lat)
        roof_sunshine = gpd.clip(roof_sunshine, bounds_box)
        if roof_sunshine.empty:
            print(f"Warning: Sunshine grid clipped to empty after bounds application for rooftop shadows.")
            roof_sunshine = gpd.GeoDataFrame(
                geometry=[], crs="EPSG:4326"
            )
            roof_sunshine['sunshine'] = pd.Series(dtype=float)
            roof_sunshine['sunshine_minutes_in_hour'] = pd.Series(dtype=float)

    # Step 5: Define the specific hour
    specific_time_start = datetime.strptime(f"{date} {hour}:00", '%Y-%m-%d %H:%M')
    specific_time_start = pytz.timezone('Asia/Hong_Kong').localize(specific_time_start)
    
    # Check solar altitude to determine if it's a daylight hour (for initial viz mode, but we always compute)
    centroid_lat = buildings_analysis.geometry.centroid.y.mean() if not buildings_analysis.empty else (lat1 + lat2) / 2
    centroid_lon = buildings_analysis.geometry.centroid.x.mean() if not buildings_analysis.empty else (lon1 + lon2) / 2
    altitude = get_altitude(centroid_lat, centroid_lon, specific_time_start.astimezone(pytz.UTC))

    # Step 6: Initialize new columns for sunshine minutes
    for sunshine in [ground_sunshine, roof_sunshine]:
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
            is_sunlit_ground = pd.Series([False] * len(ground_sunshine), index=ground_sunshine.index)
            is_sunlit_roof = pd.Series([False] * len(roof_sunshine), index=roof_sunshine.index)
            if verbose:
                print(f"Time: {t}, No sun (altitude {minute_altitude:.2f}°), no sunshine")
        else:
            try:
                if not buildings_analysis.empty:
                    shadows_ground = pybdshadow.bdshadow_sunlight(buildings_analysis, t, roof=False, include_building=False)
                    shadows_roof = pybdshadow.bdshadow_sunlight(buildings_analysis, t, roof=True, include_building=False)
                else:
                    shadows_ground = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                    shadows_roof = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
                
                # Process ground shadows
                if shadows_ground.empty:
                    is_sunlit_ground = pd.Series([True] * len(ground_sunshine), index=ground_sunshine.index)
                else:
                    # Fix invalid geometries before union
                    shadows_ground.geometry = shadows_ground.geometry.apply(make_valid)
                    # Filter out invalid, empty, or degenerate (zero-area) geometries to prevent topology errors
                    shadows_ground = shadows_ground[
                        shadows_ground.geometry.is_valid & 
                        (~shadows_ground.geometry.is_empty) & 
                        (shadows_ground.geometry.area > 1e-10)  # Tiny threshold for zero-area
                    ].copy()  # .copy() to avoid SettingWithCopyWarning
                    
                    if shadows_ground.empty:
                        is_sunlit_ground = pd.Series([True] * len(ground_sunshine), index=ground_sunshine.index)
                    else:
                        try:
                            shadow_union_ground = shadows_ground.union_all()
                            is_sunlit_ground = ~ground_sunshine.geometry.intersects(shadow_union_ground)
                        except Exception as union_err:  # Catch any remaining union failures
                            print(f"Warning: Union failed for ground ({union_err}), falling back to per-geometry intersects.")
                            # Fallback: Check intersects with each shadow individually (slower but robust)
                            is_sunlit_ground = pd.Series([True] * len(ground_sunshine), index=ground_sunshine.index)
                            for _, shadow in shadows_ground.iterrows():
                                is_sunlit_ground &= ~ground_sunshine.geometry.intersects(shadow.geometry)

                # Process roof shadows - FIXED VERSION
                if shadows_roof.empty:
                    is_sunlit_roof = pd.Series([True] * len(roof_sunshine), index=roof_sunshine.index)
                else:
                    # Fix invalid geometries before union
                    shadows_roof.geometry = shadows_roof.geometry.apply(make_valid)
                    # Filter out invalid, empty, or degenerate (zero-area) geometries to prevent topology errors
                    shadows_roof = shadows_roof[
                        shadows_roof.geometry.is_valid & 
                        (~shadows_roof.geometry.is_empty) & 
                        (shadows_roof.geometry.area > 1e-10)  # Tiny threshold for zero-area
                    ].copy()
                    
                    if shadows_roof.empty:
                        is_sunlit_roof = pd.Series([True] * len(roof_sunshine), index=roof_sunshine.index)
                    else:
                        try:
                            # CRITICAL FIX: For rooftop analysis, we need to handle self-shadowing differently
                            # Rooftop points should only be in shadow if they fall within OTHER buildings' rooftop shadows
                            shadow_union_roof = shadows_roof.union_all()
                            
                            # Create a mask for points that are actually on building rooftops
                            building_polygons = buildings_analysis.unary_union
                            points_on_buildings = roof_sunshine.geometry.intersects(building_polygons)
                            
                            # Points on buildings should only be shaded if they intersect rooftop shadows
                            # Points NOT on buildings (ground) should not be considered in rooftop analysis
                            is_sunlit_roof = pd.Series([True] * len(roof_sunshine), index=roof_sunshine.index)
                            
                            # Only apply shadow intersection to points that are actually on building rooftops
                            rooftop_points_mask = points_on_buildings
                            if rooftop_points_mask.any():
                                shadow_intersects = roof_sunshine[rooftop_points_mask].geometry.intersects(shadow_union_roof)
                                is_sunlit_roof.loc[rooftop_points_mask] = ~shadow_intersects
                                
                        except Exception as union_err:
                            print(f"Warning: Union failed for roof ({union_err}), falling back to per-geometry intersects.")
                            # Fallback with improved logic
                            is_sunlit_roof = pd.Series([True] * len(roof_sunshine), index=roof_sunshine.index)
                            building_polygons = buildings_analysis.unary_union
                            points_on_buildings = roof_sunshine.geometry.intersects(building_polygons)
                            
                            for _, shadow in shadows_roof.iterrows():
                                if shadow.geometry.is_valid and not shadow.geometry.is_empty:
                                    shadow_intersects = roof_sunshine[points_on_buildings].geometry.intersects(shadow.geometry)
                                    is_sunlit_roof.loc[points_on_buildings] &= ~shadow_intersects
                    
                if verbose:
                    print(f"Time: {t}, Shadows calculated (altitude {minute_altitude:.2f}°)")
            except ValueError as e:
                if "Given time before sunrise or after sunset" in str(e):
                    # Fallback: treat as no sun
                    is_sunlit_ground = pd.Series([False] * len(ground_sunshine), index=ground_sunshine.index)
                    is_sunlit_roof = pd.Series([False] * len(roof_sunshine), index=roof_sunshine.index)
                    if verbose:
                        print(f"Time: {t}, Library error (likely no sun), no sunshine")
                else:
                    raise e  # Re-raise if not the expected error
        
        # Add sunshine only if sunlit
        ground_sunshine.loc[is_sunlit_ground, 'sunshine_minutes_in_hour'] += 1
        roof_sunshine.loc[is_sunlit_roof, 'sunshine_minutes_in_hour'] += 1

    # Step 8: Visualize ground, roof, and combined
    has_sunshine_ground = ground_sunshine['sunshine_minutes_in_hour'].max() > 0 if not ground_sunshine.empty else False
    has_sunshine_roof = roof_sunshine['sunshine_minutes_in_hour'].max() > 0 if not roof_sunshine.empty else False
    has_sunshine = has_sunshine_ground or has_sunshine_roof

    # Compute figsize based on bounds
    figsize = compute_figsize(min_lon, max_lon, min_lat, max_lat)

    # Ground visualization (original geopandas plot)
    fig_ground, ax_ground = plt.subplots(1, 1, figsize=figsize)
    if has_sunshine_ground:
        ground_sunshine.plot(
            ax=ax_ground,
            column='sunshine_minutes_in_hour',
            cmap='plasma',
            alpha=1,
            norm=colors.Normalize(vmin=0, vmax=60),
            legend=True,
            legend_kwds={'label': "Sunshine Minutes (Ground)", 'orientation': "vertical", 'shrink': 0.8, 'pad': 0.02}
        )
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax_ground, edgecolor='k', facecolor=(0, 0, 0, 0))
    else:
        # Plot only building shapes if no sunshine in hour
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax_ground, facecolor='none', edgecolor='black', linewidth=0.5)
        ax_ground.set_title(f'Building Shapes - Hour {hour}:00 (No Sunlight, Ground), {date}')
    
    title_str_ground = f'Sunshine Minutes in Hour ({hour}:00-{hour+1}:00, {date}, Ground)' if has_sunshine_ground else f'Building Shapes - Hour {hour}:00 (No Sunlight, Ground), {date}'
    ax_ground.set_title(title_str_ground, fontsize=12)
    
    # Set STRICT bounds (no buffer)
    ax_ground.set_xlim(min_lon, max_lon)
    ax_ground.set_ylim(min_lat, max_lat)
    ax_ground.set_aspect('equal')

    # Adjust axis labels font size
    ax_ground.set_xlabel('Longitude', fontsize=8)
    ax_ground.set_ylabel('Latitude', fontsize=8)

    # Adjust tick font size
    ax_ground.tick_params(axis='both', labelsize=9)

    # Adjust colorbar tick and label font sizes after plotting
    if has_sunshine_ground:
        cbar_ground = ax_ground.get_legend()  # For geopandas plot legend
        if cbar_ground:
            cbar_ground.set_label('Sunshine Minutes (Ground)', fontsize=10)
            for t in cbar_ground.get_ticks():
                t.label.set_fontsize(10)

    # Save ground visualization
    base_path = os.path.splitext(file_path)[0]
    ground_path = f"{base_path}_{timestamp}_sunshine_ground_hour_{hour:02d}.png"
    plt.figure(fig_ground.number)
    plt.savefig(ground_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Ground sunshine map for hour {hour:02d} saved as: {ground_path}")

    # Roof visualization (original geopandas plot)
    fig_roof, ax_roof = plt.subplots(1, 1, figsize=figsize)
    if has_sunshine_roof:
        roof_sunshine.plot(
            ax=ax_roof,
            column='sunshine_minutes_in_hour',
            cmap='plasma',
            alpha=1,
            norm=colors.Normalize(vmin=0, vmax=60),
            legend=True,
            legend_kwds={'label': "Sunshine Minutes (Rooftop)", 'orientation': "vertical", 'shrink': 0.8, 'pad': 0.02}
        )
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax_roof, edgecolor='k', facecolor=(0, 0, 0, 0))
    else:
        # Plot only building shapes if no sunshine in hour
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax_roof, facecolor='none', edgecolor='black', linewidth=0.5)
        ax_roof.set_title(f'Building Shapes - Hour {hour}:00 (No Sunlight, Rooftop), {date}')
    
    title_str_roof = f'Sunshine Minutes in Hour ({hour}:00-{hour+1}:00, {date}, Rooftop)' if has_sunshine_roof else f'Building Shapes - Hour {hour}:00 (No Sunlight, Rooftop), {date}'
    ax_roof.set_title(title_str_roof, fontsize=12)
    
    # Set STRICT bounds (no buffer)
    ax_roof.set_xlim(min_lon, max_lon)
    ax_roof.set_ylim(min_lat, max_lat)
    ax_roof.set_aspect('equal')

    # Adjust axis labels font size
    ax_roof.set_xlabel('Longitude', fontsize=8)
    ax_roof.set_ylabel('Latitude', fontsize=8)

    # Adjust tick font size
    ax_roof.tick_params(axis='both', labelsize=9)

    # Adjust colorbar tick and label font sizes after plotting
    if has_sunshine_roof:
        cbar_roof = ax_roof.get_legend()  # For geopandas plot legend
        if cbar_roof:
            cbar_roof.set_label('Sunshine Minutes (Rooftop)', fontsize=10)
            for t in cbar_roof.get_ticks():
                t.label.set_fontsize(10)

    # Save roof visualization
    roof_path = f"{base_path}_{timestamp}_sunshine_roof_hour_{hour:02d}.png"
    plt.figure(fig_roof.number)
    plt.savefig(roof_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Roof sunshine map for hour {hour:02d} saved as: {roof_path}")

    # Combined visualization (direct geopandas plot on concatenated GDF)
    combined_fig = None
    if not ground_sunshine.empty and not roof_sunshine.empty:
        # Concatenate the two GeoDataFrames into a single one for unified visualization
        ground_sunshine['type'] = 'ground'
        roof_sunshine['type'] = 'roof'
        combined_sunshine = pd.concat([ground_sunshine, roof_sunshine], ignore_index=True)
        # Explicitly set CRS after concatenation to avoid warnings
        combined_sunshine.crs = 'EPSG:4326'
        
        # Compute figsize based on bounds
        figsize = compute_figsize(min_lon, max_lon, min_lat, max_lat)
        
        # Create figure with white background explicitly
        combined_fig, ax_combined = plt.subplots(1, 1, figsize=figsize, facecolor='white')
        ax_combined.set_facecolor('white')
        
        # Direct plot on combined GDF to preserve exact point coverage and white gaps
        has_sunshine_combined = combined_sunshine['sunshine_minutes_in_hour'].max() > 0
        if has_sunshine_combined:
            combined_sunshine.plot(
                ax=ax_combined,
                column='sunshine_minutes_in_hour',
                cmap='plasma',
                alpha=1,
                norm=colors.Normalize(vmin=0, vmax=60),
                legend=True,
                legend_kwds={'label': "Sunshine Minutes (Combined)", 'orientation': "vertical", 'shrink': 0.8, 'pad': 0.02}
            )
            if not buildings_analysis.empty:
                buildings_analysis.plot(ax=ax_combined, edgecolor='k', facecolor=(0, 0, 0, 0))
        else:
            # No sunshine: just buildings
            if not buildings_analysis.empty:
                buildings_analysis.plot(ax=ax_combined, facecolor='none', edgecolor='black', linewidth=0.5)
            ax_combined.set_title(f'Building Shapes - Hour {hour}:00 (No Sunlight, Combined), {date}')
        
        title_str_combined = f'Combined Ground & Rooftop Sunshine in Hour ({hour}:00-{hour+1}:00, {date})' if has_sunshine_combined else f'Building Shapes - Hour {hour}:00 (No Sunlight, Combined), {date}'
        ax_combined.set_title(title_str_combined, fontsize=12)
        
        # Set STRICT limits (no margin, no extra space)
        ax_combined.set_xlim(min_lon, max_lon)
        ax_combined.set_ylim(min_lat, max_lat)
        ax_combined.set_aspect('equal')
        ax_combined.set_xlabel('Longitude', fontsize=8)
        ax_combined.set_ylabel('Latitude', fontsize=8)
        ax_combined.tick_params(axis='both', labelsize=9)
        
        # Adjust colorbar if present
        if has_sunshine_combined:
            cbar_combined = ax_combined.get_legend()  # For geopandas plot legend
            if cbar_combined:
                cbar_combined.set_label('Sunshine Minutes (Combined)', fontsize=10)
                for t in cbar_combined.get_ticks():
                    t.label.set_fontsize(10)
        
        plt.tight_layout()

        # Save combined visualization
        combined_path = f"{base_path}_{timestamp}_sunshine_combined_hour_{hour:02d}.png"
        plt.figure(combined_fig.number)
        plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Combined sunshine map for hour {hour:02d} saved as: {combined_path}")

    if show_plot:
        if combined_fig:
            plt.show()
        else:
            # Show individual if no combined
            plt.show()
    else:
        if combined_fig:
            plt.close(combined_fig)
        plt.close(fig_ground)
        plt.close(fig_roof)

    # Step 10: Print summary statistics (only if has sunshine)
    if has_sunshine:
        print("Ground sunshine statistics:")
        print(ground_sunshine['sunshine_minutes_in_hour'].describe())
        print("Rooftop sunshine statistics:")
        print(roof_sunshine['sunshine_minutes_in_hour'].describe())
    
    # Return all three figures in a dictionary
    figures_dict = {
        'ground': fig_ground,
        'roof': fig_roof,
        'combined': combined_fig
    }
    
    return ground_sunshine, roof_sunshine, figures_dict


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
    
    # Concatenate the two GeoDataFrames into a single one for unified visualization
    ground_sunshine['type'] = 'ground'
    roof_sunshine['type'] = 'roof'
    combined_sunshine = pd.concat([ground_sunshine, roof_sunshine], ignore_index=True)
    # Explicitly set CRS after concatenation to avoid warnings
    combined_sunshine.crs = 'EPSG:4326'
    
    # Define strict bounds
    min_lon = min_lon
    max_lon = max_lon
    min_lat = min_lat
    max_lat = max_lat
    
    # Compute figsize based on bounds
    figsize = compute_figsize(min_lon, max_lon, min_lat, max_lat)
    
    # Create figure with white background explicitly
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    # Direct plot on combined GDF to preserve exact point coverage and white gaps
    has_sunshine_combined = combined_sunshine['sunshine_minutes_in_hour'].max() > 0
    if has_sunshine_combined:
        combined_sunshine.plot(
            ax=ax,
            column='sunshine_minutes_in_hour',
            cmap='plasma',
            alpha=1,
            norm=colors.Normalize(vmin=0, vmax=60),
            legend=True,
            legend_kwds={'label': "Sunshine Minutes (0-60)", 'orientation': "vertical", 'shrink': 0.8, 'pad': 0.02}
        )
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax, edgecolor='k', facecolor=(0, 0, 0, 0))
    else:
        # No sunshine: just buildings
        if not buildings_analysis.empty:
            buildings_analysis.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)
        ax.set_title(f'Building Shapes - Hour {hour}:00 (No Sunlight, Combined), {date}')
    
    title_str = f'Combined Ground & Rooftop Sunshine\nHour {hour:02d}:00-{hour+1:02d}:00, {date}' if has_sunshine_combined else f'Building Shapes - Hour {hour}:00 (No Sunlight, Combined), {date}'
    ax.set_title(title_str, fontsize=12)
    
    # Set STRICT limits (no margin, no extra space)
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    
    # Adjust colorbar if present
    if has_sunshine_combined:
        cbar = ax.get_legend()  # For geopandas plot legend
        if cbar:
            cbar.set_label('Sunshine Minutes (0-60)', fontsize=10)
            for t in cbar.get_ticks():
                t.label.set_fontsize(10)
    
    plt.tight_layout()
    
    # Save the combined overlay figure with white background
    combined_path = f"{base_path}_{timestamp}_combined_overlay_sunshine_hour_{hour:02d}.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Combined overlay sunshine map for hour {hour:02d} saved as: {combined_path}")
    
    plt.show()
    return fig