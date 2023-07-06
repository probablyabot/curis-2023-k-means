import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Function to create adversarial dataset
def create_adversarial_dataset():
    # Generate random data points with clusters
    X, y = make_blobs(n_samples=500, centers=5, random_state=42)

    # Add outliers
    outliers = np.array([[10, -5], [-12, 0], [15, 12]])
    X = np.concatenate((X, outliers))

    # Uneven cluster sizes
    small_cluster = np.array([[-5, 10], [-6, 8]])
    X = np.concatenate((X, small_cluster))

    # Overlapping clusters
    overlapping_cluster = np.array([[2, 4], [3, 3]])
    X = np.concatenate((X, overlapping_cluster))

    # Non-convex clusters
    non_convex_cluster = np.array([[-4, 4], [4, -4]])
    X = np.concatenate((X, non_convex_cluster))

    # Different cluster densities
    dense_cluster = np.array([[5, 5], [6, 6], [7, 7]])
    X = np.concatenate((X, dense_cluster))

    # Cluster shape bias
    biased_cluster = np.array([[0, 10], [2, 12], [4, 14]])
    X = np.concatenate((X, biased_cluster))

    # Add noise
    noise = np.random.rand(50, 2) * 30 - 15
    X = np.concatenate((X, noise))

    return X

# Generate adversarial dataset
X = create_adversarial_dataset()

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Adversarial Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

