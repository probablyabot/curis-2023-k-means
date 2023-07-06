import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix
import random
import math
import gurobipy as gp
from gurobipy import GRB



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
    m = gp.Model("k-means")
    n = len(points)

    D = np.square(distance_matrix(points, points))
    Z = m.addMVar(shape=(n,n), vtype=GRB.BINARY)
    y = Z.sum(axis=0)



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

data = np.array([
    [0, 0],
    [1, 0],
    [1 + np.cos(2.*np.pi/5.), np.sin(2.*np.pi/5.)],
    [0.5, 0.5 * np.tan(2.*np.pi/5.)],
    [-np.cos(2.*np.pi/5.), np.sin(2.*np.pi/5.)]
])
print(data)
k = 2

m, cost = sdp_k_means(data, k)
opt =
print('SDP solver returned matrix: \n', np.around(m, 3))
print('Objective function value: ', round(cost, 3))


