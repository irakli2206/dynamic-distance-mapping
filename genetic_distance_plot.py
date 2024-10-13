import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from get_distances_data import valid_names
from get_distances_data import distances
from matplotlib.colors import LinearSegmentedColormap

# Read the shapefile
world = gpd.read_file('gaul1_asap.shp')

# Handle encoding issues for region names
# (Assuming 'name1' is the column that contains the region names in your shapefile)

# Example genetic distance data (replace with the correct names if needed)
genetic_data = {
    "Distance": distances,
    "Region": valid_names
}

# Add a column for genetic distances
world['genetic_distance'] = world['name1'].map(dict(zip(genetic_data['Region'], genetic_data['Distance'])))

# Print the first few entries of genetic_distance
# print('All entries of names:\n', world['name1'].unique().tolist())
# Filter for Poland
country = 'France'
regions = world[world['name0'] == country]
region_names = regions['name1'].unique().tolist()
print('All entries of names for' + country + '\n', region_names)

# Check how many regions have NaN distances
nan_count = world['genetic_distance'].isna().sum()
print(f"Regions with NaN genetic distances: {nan_count}")

# Filter out rows where genetic_distance is NaN
world_filtered = world[pd.notna(world['genetic_distance'])].copy()

# Create a custom colormap from green to red
cmap = LinearSegmentedColormap.from_list('custom_green_red', [(0.0, 1.0, 0.0), 'red'])

# Update normalization for color mapping
norm = plt.Normalize(vmin=min(distances), vmax=0.2)

# Set color based on genetic distance using .loc to avoid the warning
world_filtered.loc[:, 'color'] = world_filtered['genetic_distance'].apply(
    lambda x: cmap(norm(x))
)

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
