import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from get_distances_data import valid_names
from get_distances_data import distances
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

# Read the shapefile
world = gpd.read_file('gaul1_asap.shp')

# Example genetic distance data
genetic_data = {
    "Distance": distances,
    "Region": valid_names
}

# Add a column for genetic distances
world['genetic_distance'] = world['name1'].map(dict(zip(genetic_data['Region'], genetic_data['Distance'])))

# Print the first few entries of genetic_distance
# print('All entries of names:\n', world['name0'].unique().tolist())

# Filter for a specific country (e.g., Tunisia)
country = 'Iran (Islamic Republic of)'
regions = world[world['name0'] == country]
region_names = regions['name1'].unique().tolist()
print(f'All entries of names for {country}:\n', region_names)

# Check how many regions have NaN distances
nan_count = world['genetic_distance'].isna().sum()
print(f"Regions with NaN genetic distances: {nan_count}")

# Separate regions with and without genetic distance
regions_with_distance = world.dropna(subset=['genetic_distance'])
regions_without_distance = world[world['genetic_distance'].isna()]

# Interpolate missing genetic distances using nearest neighbor
# if not regions_without_distance.empty:
#     # Get the centroids of regions with valid distances and their genetic distances
#     valid_coords = np.array(list(regions_with_distance.geometry.centroid.apply(lambda geom: (geom.x, geom.y))))
#     valid_distances = regions_with_distance['genetic_distance'].values
    
#     # Build a KDTree for fast nearest-neighbor lookup
#     kdtree = cKDTree(valid_coords)
    
#     # Find nearest neighbors for regions with missing distances
#     missing_coords = np.array(list(regions_without_distance.geometry.centroid.apply(lambda geom: (geom.x, geom.y))))
#     _, nearest_indices = kdtree.query(missing_coords)
    
#     # Assign the nearest genetic distance to the missing regions
#     world.loc[world['genetic_distance'].isna(), 'genetic_distance'] = valid_distances[nearest_indices]

# Check the updated NaN count
nan_count_after = world['genetic_distance'].isna().sum()
print(f"Regions with NaN genetic distances after interpolation: {nan_count_after}")

# Filter out rows where genetic_distance is NaN (in case there are any remaining)
world_filtered = world[pd.notna(world['genetic_distance'])].copy()

# Create a custom colormap from green to red
cmap = LinearSegmentedColormap.from_list('custom_green_red', [(0.0, 1.0, 0.0), 'red'])

# Update normalization for color mapping
norm = plt.Normalize(vmin=min(distances), vmax=0.3)

# Set color based on genetic distance using .loc to avoid the warning
world_filtered.loc[:, 'color'] = world_filtered['genetic_distance'].apply(
    lambda x: cmap(norm(x))
)

# Find the region with the lowest genetic distance
closest_population_index = world_filtered['genetic_distance'].idxmin()
closest_population_name = world_filtered.loc[closest_population_index, 'name1']
closest_population_distance = world_filtered.loc[closest_population_index, 'genetic_distance']

# Define the boundaries for Europe
xlim = (-30, 50)  # Longitude limits for Europe
ylim = (35, 70)   # Latitude limits for Europe

# Create and configure the figure
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.boundary.plot(ax=ax, linewidth=0.5, color='k')

# Plot only regions with valid genetic distances
world_filtered.plot(ax=ax, color=world_filtered['color'], legend=False)

ax.set_xlim(xlim)
ax.set_ylim(ylim)

# Add a color bar to represent the genetic distances
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('Genetic Distance')

# Add titles and labels
target_name = "Georgia_Kotias_CHG"
plt.title(f"Genetic Distances to {target_name}")
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add the closest population to the legend
plt.legend([f'Closest Population: {closest_population_name}, Distance: {closest_population_distance:.4f}'], loc='upper left')

# Use tight_layout for better fit
plt.tight_layout()

# Save the figure with improved quality
plt.savefig('genetic_distance_map.png', dpi=300, bbox_inches='tight')  # Save as PNG with high DPI

plt.show()
