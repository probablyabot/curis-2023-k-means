import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from time import time


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
# time; doesn't scale for large n.
def optimal_k_means(points, k):
    m = gp.Model()
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


# Samples `num_points` points from a d-dimensional ball with the given radius
# and center, returning an array of shape (num_points, d).
def sample_from_ball(num_points, d=2, radius=1, center=None):
    rng = np.random.default_rng()
    if center is None:
        center = np.zeros(d)
    if len(center) != d:
        return -1
    r = rng.random(num_points) ** (1/d)
    theta = rng.normal(size=(d, num_points))
    theta /= np.linalg.norm(theta, axis=0)
    return center + radius * (theta * r).T


num_points = 25
data = sample_from_ball(num_points)
data = np.concatenate([data, sample_from_ball(num_points, center=(50, 50))])


# Generate 2D coordinates for a regular n-gon inscribed in a circle of given
# radius centered at the origin.
def get_polygon_coordinates(n, radius=1):
    coordinates = []
    theta = 2 * np.pi / n

    for i in range(n):
        x = radius * np.cos(theta * i)
        y = radius * np.sin(theta * i)
        coordinates.append([x, y])

    return np.array(coordinates)


# data = get_polygon_coordinates(5, radius=1)

# data = np.array([
#     [0, 0],
#     [1, 0],
#     [0.5, np.sin(np.pi / 3.)]
# ])
print('Input data:\n', data)
k = 2

start_time = time()
m, cost = sdp_k_means(data, k)
print('SDP solver returned matrix:\n', np.around(m, 3))
print('SDP objective function value:', round(cost, 3))
print('SDP running time:', time() - start_time, 'seconds')
# start_time = time()
# opt = optimal_k_means(data, k)
# print('Optimal objective function value:', round(opt, 3))
# print('Gurobi running time: ', time() - start_time)
