import numpy as np
import cvxpy as cp
from scipy.spatial import distance_matrix
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, permutations


# Given n points and k, uses semi-definite programming to produce a solution
# to the (relaxed) k-means clustering problem.
def sdp_k_means(points, k, psd=True, tri=False, l=0.0):
    n = points.shape[0]

    D = np.square(distance_matrix(points, points))
    M = cp.Variable((n, n), PSD=psd)
    if l:
        obj = cp.Minimize(cp.trace(D.T @ M) / 2 + l * cp.trace(M))
    else:
        obj = cp.Minimize(0.5 * cp.trace(D.T @ M))

    constraints = [cp.sum(M, axis=0) >= 1]
    for i in range(n):
        constraints.append(M[i, i] >= M[i])
    if k:
        constraints += [cp.trace(M) <= k]
    if tri:
        for i, j, m in combinations(range(n), 3):
            for pi, pj, pm in [(i, j, m), (j, m, i), (m, i, j)]:
                constraints += [M[pj, pj] + M[pi, pm] >= M[pi, pj] + M[pj, pm]]
    constraints += [M >= 0]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    return M.value, obj.value, [c.dual_value for c in constraints]


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
    m.addConstrs(Y[j] >= M[i][j] for i in range(n) for j in range(s))
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
        if ',' not in pt_raw:
            return [float(pt_raw)] * expected_d
        pt = [float(x) for x in pt_raw.split(',')]
        if len(pt) == expected_d:
            return pt
        print(f'Invalid point, expected dimension {expected_d}')


# Calculates optimal objective value for well-separated regular n-gons by
# taking consecutive points as clusters.
def optimal_polygon(points, n, k):
    nc = points.shape[0] // n
    obj = 0
    for i in range(nc):
        ki = (i + 1) * k // nc - i * k // nc
        clusters = [points[i*n+j*n//ki:i*n+(j+1)*n//ki] for j in range(ki)]
        centroids = np.array([np.mean(clusters[j], axis=0) for j in range(ki)])
        obj += np.sum([np.sum((clusters[i] - centroids[i]) ** 2) for i in range(ki)])
    return obj


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


def gen_simplex(n, center, size=1):
    pts = []
    for i in range(n):
        pts.append([0] * i + [size] + [0] * (n - i - 1))
    return center + np.array(pts)


def gen_simplex_clusters(num_clusters, n, centers, size=1):
    simplices = [gen_simplex(n, centers[i], size) for i in range(num_clusters)]
    return np.vstack(simplices)


def optimal_simplex(points, n, k):
    nc = points.shape[0] // n
    obj = 0
    for i in range(nc):
        ki = (i + 1) * k // nc - i * k // nc
        clusters = [points[i*n+j*n//ki:i*n+(j+1)*n//ki] for j in range(ki)]
        centroids = np.array([np.mean(clusters[j], axis=0) for j in range(ki)])
        obj += np.sum([np.sum((clusters[i] - centroids[i]) ** 2) for i in range(ki)])
    return obj


def lagrangian_polygon_cost(n):
    pts = gen_polygon(n, n / (2 * np.pi), 0, 0)
    k = int(6 ** (-1/3) * n ** (2/3))
    c = optimal_polygon(pts, n, k)
    return c + n * k


def test(n):
    # print('dual cost:', n ** 2)
    lagrange = lagrangian_polygon_cost(n)
    print('UFL cost:', lagrange)
    print('optimal primal lp cost:', 2 * n)
    # print('dual/UFL ratio:', n ** 2 / lagrange)
    print('UFL/primal ratio:', lagrange / (2 * n))


def polygon_dual(n):
    pts = gen_polygon(n, n / (2 * np.pi), 0, 0)

    a = cp.Variable()
    b = cp.Variable((n, n))
    d = [cp.Variable((n, n)) for _ in range(n)]

    constraints = [b >= 0]
    constraints += [cp.diag(b) == 0]
    constraints += [d[i] >= 0 for i in range(n)]
    for i in range(n):
        constraints += [cp.diag(d[i]) == 0]
        constraints += [d[i][i] == 0]
        constraints += [d[i][:, i] == 0]
    for i in range(n):
        constraints.append(a + cp.sum(b[i]) + cp.sum(d[i]) <= n)
    for i in range(n-1):
        for j in range(i+1, n):
            constraints.append(b[j][i] == 0)
            c = 2 * a - 2 * b[i][j]
            for k in range(n):
                if k != i and k != j:
                    c += d[k][i][j]
                    constraints.append(d[k][j][i] == 0)
                    k1, k2 = sorted([i, k])
                    c -= d[j][k1][k2]
                    constraints.append(d[j][k2][k1] == 0)
                    k1, k2 = sorted([j, k])
                    c -= d[i][k1][k2]
                    constraints.append(d[i][k2][k1] == 0)
            constraints.append(c <= np.linalg.norm(pts[i] - pts[j]) ** 2)

    prob = cp.Problem(cp.Maximize(a), constraints)
    prob.solve()

    return a.value, b.value, [d[i].value for i in range(n)]


n = 12
a, b, d = polygon_dual(n)
print('alpha:', np.round(a, 3))
print('beta:')
print(np.round(b, 3))
for i in range(n):
    print(f'delta[{i}]:')
    print(np.round(d[i], 3))


# for e in range(4, 10):
#     n = 2 ** e
#     test(n)

# for i in range(3, 8):
#     n = 2 ** i
#     points = gen_polygon(n, 1, 0, 0)
    # print(f'ratio for {n}:', optimal_polygon(n, 1, 2) / construct_lp(points, 2))
    # print(f'ratio for {n}:', optimal_polygon(n, 1, 3) / sdp_k_means(points, 3, psd=False, tri=True)[1])
    # print(f'optimal LP solution for n={n}:', sdp_k_means(points, 3, psd=False)[1])
    # print(f'constructed LP solution for n={n}:', construct_lp(points, 3))
