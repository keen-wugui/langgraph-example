import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.5, x.size)

# Create the plot
fig, ax = plt.subplots()

# Plot the data with artistic styling
ax.plot(x, y, linestyle='-', marker='o', color='teal', markersize=5, markerfacecolor='orange', markeredgewidth=2, markeredgecolor='purple')

# Add a grid with custom styling
ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# Add a random title and axis labels
ax.set_title('Whimsical Waves of Randomness', fontsize=16, fontweight='bold', color='navy')
ax.set_xlabel('Time (s)', fontsize=14, fontstyle='italic', color='darkred')
ax.set_ylabel('Amplitude', fontsize=14, fontstyle='italic', color='darkred')

# Customize the ticks
ax.tick_params(axis='x', colors='green', direction='inout', length=6, width=2)
ax.tick_params(axis='y', colors='blue', direction='inout', length=6, width=2)

# Add a background color
fig.patch.set_facecolor('lavender')

# Show the plot
plt.show()