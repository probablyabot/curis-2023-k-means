import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix

def k_means_clustering(points, k):
    n = len(points)
    d = len(points[0])

    D = np.square(distance_matrix(points, points))
    M = cp.Variable((n, n), symmetric=True)

    obj = cp.Minimize(cp.sum(cp.multiply(D, M), axis=None))

    constraints = [cp.sum(M, axis=0) == 1, cp.sum(M, axis=1) == 1, M >> 0]
    constraints += [cp.trace(M) == k]
    for i in range(n):
        constraints += [M[i, i] >= M[i, j] for j in range(n)]
        constraints += [M[i, i] >= M[j, i] for j in range(n)]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    return M.value, obj.value

points = np.array([[0,0], [100,100]])
k = 2

m, cost = k_means_clustering(points, k)
print('SDP solver returned matrix: ', m)
print('Objective function value: ', cost)