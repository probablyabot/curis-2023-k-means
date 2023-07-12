import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from time import time
import argparse
from typing import Tuple


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
def sample_from_ball(num_points, d, radius, center):
    rng = np.random.default_rng()
    r = rng.random(num_points) ** (1/d)
    theta = rng.normal(size=(d, num_points))
    theta /= np.linalg.norm(theta, axis=0)
    return center + radius * (theta * r).T


def gen_clusters(num_clusters, points_per_cluster, d, radius, centers):
    clusters = [sample_from_ball(points_per_cluster, d, radius, centers[i])
                for i in range(num_clusters)]
    return np.vstack(clusters)


# Generate 2D coordinates for a regular n-gon inscribed in a circle of given
# radius centered at the origin.
def gen_polygon(n, radius=1):
    coordinates = []
    theta = 2 * np.pi / n

    for i in range(n):
        x = radius * np.cos(theta * i)
        y = radius * np.sin(theta * i)
        coordinates.append([x, y])

    return np.array(coordinates)


# data = np.array([
#     [0, 0],
#     [1, 0],
#     [0.5, np.sin(np.pi / 3.)]
# ])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cluster', action='store_true')
    parser.add_argument('-p', '--polygon', action='store_true')
    parser.add_argument('-m', '--manual', action='store_true')
    parser.add_argument('-nc', '--num_clusters', type=int, default=1)
    parser.add_argument('-ppc', '--pts_per_cluster', type=int, default=10)
    parser.add_argument('-d', '--dimension', type=int, default=2)
    parser.add_argument('-r', '--radius', type=float, default=1.0)
    parser.add_argument('-ns', '--num_sides', type=int, default=3)
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('-ng', '--no_gurobi', action='store_true')

    args = parser.parse_args()
    if args.cluster:
        centers = []
        for i in range(args.num_clusters):
            center = []
            while len(center) == 0:
                center_raw = input(f'Cluster center #{i}: ').replace('(', '').replace(')', '')
                center = [float(x) for x in center_raw.split(',')]
                if len(center) != args.dimension:
                    center = []
            centers.append(center)
        data = gen_clusters(args.num_clusters, args.pts_per_cluster,
                            args.dimension, args.radius, centers)
    elif args.polygon:
        data = gen_polygon(args.num_sides, args.radius)
    else:
        pts = []
        while True:
            pt = input(f'Data point #{len(pts)+1} (return to stop): ').replace('(', '').replace(')', '')
            if pt == '':
                break
            pts.append([float(x) for x in pt.split(',')])
        data = np.array(pts)
    k = args.k
    print('Input data:\n', data)

    start_t = time()
    m, cost = sdp_k_means(data, k)
    sdp_t = time()
    if not args.no_gurobi:
        opt = optimal_k_means(data, k)
        opt_t = time()
    print('SDP solver returned matrix:\n', np.around(m, 3))
    print('SDP objective function value:', round(cost, 3))
    print(f'SDP running time: {sdp_t - start_t} seconds')
    if not args.no_gurobi:
        print('Optimal objective function value:', round(opt, 3))
        print(f'Gurobi running time: {opt_t - sdp_t} seconds')
