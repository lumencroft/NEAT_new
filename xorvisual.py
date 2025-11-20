import matplotlib.pyplot as plt
import numpy as np
import datasets

X_data, y_data = datasets.make_xor_min(n=1000, noise=0.1)

plt.figure(figsize=(8, 6))
plt.scatter(X_data[y_data == 0, 0], X_data[y_data == 0, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X_data[y_data == 1, 0], X_data[y_data == 1, 1], c='red', label='Class 1', alpha=0.6)

plt.title('XOR Dataset Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.savefig('xor_data_visualization.png')

print("XOR data visualization saved to xor_data_visualization.png")