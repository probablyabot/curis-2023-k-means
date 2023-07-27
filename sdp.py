import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, permutations


# Given n points and k, uses semi-definite programming to produce a solution
# to the (relaxed) k-means clustering problem.
def sdp_k_means(points, k, psd=True):
    n = points.shape[0]

    D = np.square(distance_matrix(points, points))
    M = cp.Variable((n, n), PSD=psd)
    obj = cp.Minimize(0.5 * cp.trace(D.T @ M))

    constraints = [cp.sum(M, axis=0) >= 1]
    constraints += [cp.trace(M) <= k]
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
def optimal_k_means(points, k, centroids=None):
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()

    m = gp.Model(env=env)
    n = points.shape[0]

    # Compute all 2^n - 1 possible centroids
    if centroids is None:
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
    polygons = [gen_polygon(n, radius, *centers[i])
                for i in range(num_clusters)]
    return np.vstack(polygons)


def gen_clique_embeddings(n, k):
    numbers = [0] * (n - k) + [1] * k
    return np.vstack(list(set(permutations(numbers, n))))


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


# Calculates optimal objective value for regular n-gon by taking consecutive
# points as clusters.
def optimal_polygon(n, radius, k):
    pts = gen_polygon(n, radius, 0, 0)
    clusters = [pts[i*n//k:(i+1)*n//k] for i in range(k)]
    centroids = np.array([np.mean(clusters[i], axis=0) for i in range(k)])
    return np.sum([np.sum((clusters[i] - centroids[i]) ** 2) for i in range(k)])


# Construct an optimal solution to the LP for a regular n-gon and return the
# objective value.
def construct_lp(points, k):
    n = points.shape[0]
    D = np.square(distance_matrix(points, points))
    m = n // k
    a, b = (m + 1) // 2, (m - 1) // 2
    extra = (1 - k / n * (a + b)) / 2
    row = [k / n] * a + [extra] + [0] * (n - a - b - 2) + [extra] + [k / n] * b
    M = np.array([np.roll(row, i) for i in range(n)])
    return np.trace(D.T @ M) / 2


for i in range(3, 8):
    n = 2 * i - 1
    points = gen_polygon(n, 1, 0, 0)
    # print(f'ratio for {n}:', optimal_polygon(n, 1, 3), construct_lp(points, 3))
    # print(f'ratio for {n}:', optimal_polygon(n, 1, 3) / sdp_k_means(points, 3)[1])
    print(f'optimal LP solution for n={n}:', sdp_k_means(points, 3, psd=False)[1])
    print(f'constructed LP solution for n={n}:', construct_lp(points, 3))
