import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix
import random
import math
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

# Given n points and k, uses semi-definite programming to produce a solution
# to the (relaxed) k-means clustering problem.
def sdp_k_means(points, k):
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


# Solves for the optimal k-means clustering by reducing to the discrete case and  using Gurobi's integer programming
# solver. Note: Doesn't scale for large n.
def optimal_k_means(points, k):
    m = gp.Model('k-means')
    n = len(points)

    centroids = []
    for c in range(1, n + 1):
        for cluster in combinations(points, c):
            centroids.append(np.mean(cluster, axis=0))
    s = len(centroids)
    
    D = np.square(distance_matrix(points, centroids))
    M = m.addMVar(shape=(n, s), vtype=GRB.BINARY)
    Y = m.addMVar(shape=(s,), vtype=GRB.BINARY)

    m.setObjective((D * M).sum(), GRB.MINIMIZE)

    m.addConstrs(M[i].sum() >= 1 for i in range(n))
    m.addConstrs((Y[j] >= M[i][j] for i in range(n) for j in range(s)))
    m.addConstr(Y.sum() <= k, 'k clusters')
    
    m.optimize()
    return m.ObjVal


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


num_points = 20
# data = np.concatenate((np.array(sample_points(num_points)), np.array(sample_points_new(num_points))))
# data = np.array(sample_points(num_points))

def get_polygon_coordinates(n, radius=1):
    coordinates = []
    angle = 2 * math.pi / n  # Calculate the angle between each vertex
    
    for i in range(n):
        x = radius * math.cos(i * angle)
        y = radius * math.sin(i * angle)
        coordinates.append((x, y))
    
    return coordinates

# Example usage
num_sides = 12  # Number of sides of the polygon
polygon_radius = 1  # Radius of the polygon
polygon_coordinates = get_polygon_coordinates(num_sides, polygon_radius)
data = np.array(polygon_coordinates)
# data = np.array([
#     (0, 0, 0),
#     (1, 0, 1),
#     (0, 1, 1)
# ])
print(np.around(data, 3))
k = 5

m, cost = sdp_k_means(data, k)
opt = optimal_k_means(data, k)
print('SDP solver returned matrix: \n', np.around(m, 3))
print('SDP objective function value: ', round(cost, 3))
print('Optimal objective function value: ', round(opt, 3))


