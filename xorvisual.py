import matplotlib.pyplot as plt
import numpy as np
import datasets

# Generate the XOR data
X_data, y_data = datasets.make_xor(n=1000, noise=0.1)

# Create the scatter plot
plt.figure(figsize=(8, 6))
# Plot points with label 0 (class 0) as blue
plt.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1], c='blue', label='Class 0', alpha=0.6)
# Plot points with label 1 (class 1) as red
plt.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1], c='red', label='Class 1', alpha=0.6)

# Add titles and labels
plt.title('XOR Dataset Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Save the plot
plt.savefig('xor_data_visualization.png')

print("XOR data visualization saved to xor_data_visualization.png")