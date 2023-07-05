import numpy as np
np.set_printoptions(threshold=np.inf)
import cvxpy as cp
from scipy.spatial import distance_matrix
import random
import math
from sklearn.datasets import make_blobs

# Given n points and k, uses semi-definite programming to produce a solution
# to the (relaxed) k-means clustering problem.
def k_means_clustering(points, k):
    n = len(points)

    D = np.square(distance_matrix(points, points))
    M = cp.Variable((n, n), PSD=True)
    obj = cp.Minimize(0.5 * cp.trace(D.T @ M))

    constraints = [cp.sum(M, axis=0) == 1, cp.sum(M, axis=1) == 1]
    constraints += [cp.trace(M) == k]
    constraints += [M >= 0]
    for i in range(n):
        constraints += [M[i, i] >= M[i, j] for j in range(n)]
        constraints += [M[i, i] >= M[j, i] for j in range(n)]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    return M.value, obj.value

def sample_points(num_points):
    points = []
    for _ in range(num_points):
        radius = math.sqrt(random.random())  # Generate random radius between 0 and 1
        theta = random.uniform(0, 2*math.pi)  # Generate random angle between 0 and 2*pi
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        points.append((x, y))
    return points

def sample_points_new(num_points):
    points = []
    for _ in range(num_points):
        radius = math.sqrt(random.random())  # Generate random radius between 0 and 1
        theta = random.uniform(0, 2*math.pi)  # Generate random angle between 0 and 2*pi
        x = radius * math.cos(theta) + 100
        y = radius * math.sin(theta) + 100
        points.append((x, y))
    return points

# Function to create adversarial dataset
def create_adversarial_dataset():
    # Generate random data points with clusters
    X, y = make_blobs(n_samples=30, centers=5, random_state=42)

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


# Usage example
num_points = 20
# data = np.concatenate((np.array(sample_points(num_points)), np.array(sample_points_new(num_points))))
data = X
# data = np.array(sample_points(num_points))

# data = np.array([
#     [0, 0],
#     [100, 100],
# ])
k = 4

m, cost = k_means_clustering(data, k)
# print('SDP solver returned matrix: \n', m)
# print('Objective function value: ', cost)
print('SDP solver returned matrix: \n', np.around(m, 2))
print('Objective function value: ', round(cost, 3))
