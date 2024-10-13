import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from get_distances_data import valid_names
from get_distances_data import distances
from matplotlib.colors import LinearSegmentedColormap

# Read the shapefile
world = gpd.read_file('ne_10m_admin_1_states_provinces.shp')

# Handle encoding issues for region names
world['name'] = world['name'].apply(lambda x: x.encode('latin1').decode('utf-8') if isinstance(x, str) else x)

# Example genetic distance data (replace with the correct names if needed)
genetic_data = {
    "Distance": distances,
    "Region": valid_names
}

# Add a column for genetic distances
world['genetic_distance'] = world['name'].map(dict(zip(genetic_data['Region'], genetic_data['Distance'])))

# Print the first few entries of genetic_distance
print('All entries of names:\n', world['name'].unique().tolist())

# Check how many regions have NaN distances
nan_count = world['genetic_distance'].isna().sum()
print(f"Regions with NaN genetic distances: {nan_count}")

# Interpolating genetic distances based on neighbors
points = np.array(world.geometry.apply(lambda geom: geom.representative_point().coords[0]).tolist())
values = world['genetic_distance'].values

# Use all points for interpolation, including those with NaN values
valid_points = points[~np.isnan(values)]
valid_values = values[~np.isnan(values)]

# Check valid points and values
print("Valid Points Shape:", valid_points.shape)
print("Valid Values Shape:", valid_values.shape)

# If there are valid points, proceed with interpolation
if valid_points.shape[0] > 0:
    grid_x, grid_y = np.mgrid[-180:180:100j, -90:90:100j]
    grid_z = griddata(valid_points, valid_values, (grid_x, grid_y), method='linear')

    # Update world DataFrame with interpolated values
    world['interpolated_distance'] = griddata(valid_points, valid_values, points, method='linear')

    # Filter out rows where interpolated_distance is NaN
    world_filtered = world[pd.notna(world['interpolated_distance'])].copy()

    # Create a custom colormap from green to red
    cmap = LinearSegmentedColormap.from_list('custom_green_red', [(0.0, 1.0, 0.0), 'red'])  # Using bright green

    # Update normalization for color mapping
    norm = plt.Normalize(vmin=min(distances), vmax=max(distances))

    # Set color based on interpolated distance using .loc to avoid the warning
    world_filtered.loc[:, 'color'] = world_filtered['interpolated_distance'].apply(
        lambda x: cmap(norm(x))
    )

    # Define the boundaries for Europe
    xlim = (-30, 50)  # Longitude limits for Europe
    ylim = (35, 70)   # Latitude limits for Europe

    # Create and configure the figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.boundary.plot(ax=ax, linewidth=0.5, color='k')

    # Plot only regions with valid data
    world_filtered.plot(ax=ax, color=world_filtered['color'], legend=False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add a color bar to represent the genetic distances
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Genetic Distance')

    # Add titles and labels
    plt.title('Genetic Distance Mapping (Europe Only)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Use tight_layout for better fit
    plt.tight_layout()

    # Save the figure with improved quality
    plt.savefig('genetic_distance_map.png', dpi=300, bbox_inches='tight')  # Save as PNG with high DPI

    plt.show()
else:
    print("No valid points available for interpolation.")
