import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix


# Given n points and k, uses semi-definite programming to produce a solution
# to the (relaxed) k-means clustering problem.
def k_means_clustering(points, k):
    n = len(points)

    D = np.square(distance_matrix(points, points))
    M = cp.Variable((n, n), PSD=True)
    obj = cp.Minimize(0.5 * cp.trace(D.T @ M))

    constraints = [cp.sum(M, axis=0) == 1, cp.sum(M, axis=1) == 1]
    constraints += [cp.trace(M) == k]
    for i in range(n):
        constraints += [M[i, i] >= M[i, j] for j in range(n)]
        constraints += [M[i, i] >= M[j, i] for j in range(n)]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    return M.value, obj.value


data = np.array([
    [0, 0],
    [100, 100],
])
k = 2

m, cost = k_means_clustering(data, k)
print('SDP solver returned matrix: ', m)
print('Objective function value: ', cost)
