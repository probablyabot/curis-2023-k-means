import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations


# Given n points and k, uses semi-definite programming to produce a solution
# to the (relaxed) k-means clustering problem.
def sdp_k_means(points, k):
    n = points.shape[0]

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
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    m = gp.Model(env=env)
    n = points.shape[0]

    # Compute all 2^n - 1 possible centroids
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
    m.addConstr(Y.sum() <= k, '')

    m.optimize()
    return m.ObjVal


# Calculate optimal objective value by taking blocks of size `ppc` and adding
# objective values separately. (Should only be used on instances where
# clusters are sufficiently well-separated.)
def optimal_separate(points, k, ppc):
    nc = points.shape[0] // ppc
    obj = 0
    for i in range(nc):
        obj += optimal_k_means(points[i*ppc:(i+1)*ppc], 1)
    return obj


# Samples `num_points` points from a d-dimensional ball with the given radius
# and center, returning an array of shape (num_points, d).
def sample_from_ball(num_points, d, radius, center):
    rng = np.random.default_rng()
    r = rng.random(num_points) ** (1/d)
    theta = rng.normal(size=(d, num_points))
    theta /= np.linalg.norm(theta, axis=0)
    return center + radius * (theta * r).T


# Uses `sample_from_ball()` to generate given number of clusters
def gen_clusters(num_clusters, points_per_cluster, d, radius, centers):
    clusters = [sample_from_ball(points_per_cluster, d, radius, centers[i])
                for i in range(num_clusters)]
    return np.vstack(clusters)


# Generate 2D coordinates for a regular n-gon inscribed in a circle of given
# radius centered at (cx, cy).
def gen_polygon(n, radius, cx, cy):
    coordinates = []
    theta = 2 * np.pi / n

    for i in range(n):
        x = radius * np.cos(theta * i)
        y = radius * np.sin(theta * i)
        coordinates.append([x, y])

    return (cx, cy) + np.array(coordinates)


def gen_polygon_clusters(num_clusters, n, radius, centers):
    polygons = [gen_polygon(n, radius, *centers[i]) for i in range(num_clusters)]
    return np.vstack(polygons)


# Read in a d-dimension point from input and parse it into a list of floats.
# If no input is provided, use the origin.
def parse_point(prompt, expected_d):
    print('Press return for origin.')
    while True:
        pt_raw = input(prompt).strip('()')
        if pt_raw == '':
            return [0.0] * expected_d
        pt = [float(x) for x in pt_raw.split(',')]
        if len(pt) == expected_d:
            return pt
        print(f'Invalid point, expected dimension {expected_d}')
