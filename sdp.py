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


# Solves for the optimal k-means clustering by reducing to the discrete case
# and using Gurobi's integer programming solver. Note: Exponential running
# time; doesn't scale for
# large n.
def optimal_k_means(points, k):
    m = gp.Model('k-means')
    n = len(points)

    # Compute all 2^n - 2 possible centroids
    centroids = []
    for c in range(1, n):
        for cluster in combinations(points, c):
            centroids.append(np.mean(cluster, axis=0))
    s = len(centroids)
    
    D = np.square(distance_matrix(points, centroids))
    M = m.addMVar(shape=(n, s), vtype=GRB.BINARY)
    Y = m.addMVar(shape=(s,), vtype=GRB.BINARY)

    m.setObjective((D * M).sum(), GRB.MINIMIZE)

    m.addConstrs(M[i].sum() >= 1 for i in range(n))
    m.addConstrs((Y[j] >= M[i][j] for i in range(n) for j in range(s)))
    m.addConstr(Y.sum() <= k, '')
    
    m.optimize()
    return m.ObjVal


def sample_points(num_points, cx=0, cy=0):
    points = []
    for _ in range(num_points):
        radius = math.sqrt(random.random())  # Generate random radius between 0 and 1
        theta = random.uniform(0, 2*math.pi)  # Generate random angle between 0 and 2*pi
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        points.append([x, y])
    return np.array(points)


# num_points = 20
# data = np.concatenate((sample_points(num_points), sample_points(num_points, cx=100, cy=100)))
# data = sample_points(num_points)


def get_polygon_coordinates(n, radius=1):
    coordinates = []
    angle = 2 * math.pi / n  # Calculate the angle between each vertex

    for i in range(n):
        x = radius * math.cos(i * angle)
        y = radius * math.sin(i * angle)
        coordinates.append((x, y))

    return coordinates


# Example usage
polygon_coordinates = get_polygon_coordinates(15)
data = np.array(polygon_coordinates)

# data = np.array([
#     [0, 0],
#     [1, 0],
#     [0.5, np.sin(np.pi / 3.)]
# ])
print('Input data:\n', data)
k = 4

m, cost = sdp_k_means(data, k)
opt = optimal_k_means(data, k)
print('SDP solver returned matrix:\n', np.around(m, 3))
print('SDP objective function value:', round(cost, 3))
print('Optimal objective function value:', round(opt, 3))
