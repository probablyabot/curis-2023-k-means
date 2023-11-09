import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


# function to generate an n-gon
def gen_polygon(n, r=1, cx=0, cy=0):
    t = 2 * np.pi / n
    pts = [[r*np.cos(t*i), r*np.sin(t*i)] for i in range(n)]
    return [cx, cy] + np.array(pts)


# creates and solves the dual
def ngon_dual(n):
    pts = gen_polygon(n, n / (2 * np.pi))

    a = cp.Variable()
    b = cp.Variable((n, n))
    d = [cp.Variable((n, n)) for _ in range(n)]

    constraints = [b >= 0]
    constraints += [cp.diag(b) == 0]  # beta_{ii} = 0
    constraints += [d[i] >= 0 for i in range(n)]
    for i in range(n):
        # set diagonal, i-th row, i-th col to be 0. these correspond
        # to delta_{ijj}, delta_{iij}, delta_{iji}.
        constraints += [cp.diag(d[i]) == 0]
        constraints += [d[i][i] == 0]
        constraints += [d[i][:, i] == 0]
    for i in range(n):
        # here we set lambda = n
        constraints.append(a + cp.sum(b[i]) + cp.sum(d[i]) <= n)
    for i, j, k in combinations(range(n), 3):
        # define delta_{ijk} for middle vertex i and j < k
        constraints.append(d[i][k][j] == 0)
        constraints.append(d[j][k][i] == 0)
        constraints.append(d[k][j][i] == 0)
    for i in range(n-1):
        for j in range(i+1, n):
            c = 2 * a - b[i][j] - b[j][i]
            for k in range(n):
                if k != i and k != j:
                    c += d[k][i][j]
                    f, g = sorted([i, k])
                    c -= d[j][f][g]
                    f, g = sorted([j, k])
                    c -= d[i][f][g]
            constraints.append(c <= np.linalg.norm(pts[i] - pts[j]) ** 2)

    prob = cp.Problem(cp.Maximize(a), constraints)
    prob.solve()

    return a.value, b.value, [d[i].value for i in range(n)]


# takes in an n x n numpy array and creates a heatmap
def make_heatmap(arr, cmin=None, cmax=None, title='Heatmap'):
    plt.imshow(arr, cmap='coolwarm', interpolation='nearest', vmin=cmin, vmax=cmax)
    plt.colorbar()
    plt.title(title)
    plt.xticks(np.arange(0, n, 2))
    plt.yticks(np.arange(0, n, 2))
    plt.gca().invert_yaxis()
    plt.show()


n = 20
a, b, d = ngon_dual(n)
print('alpha:', np.round(a, 3))
make_heatmap(b, cmin=0, cmax=1, title=f'betas, n={n}')
delta = np.zeros((n, n))
for i in range(1, n - 1):
    for j in range(1, n - i):
        total = 0
        for k in range(n):
            k1, k2 = sorted([(k+i) % n, (k-j) % n])
            total += d[k][k1][k2]
        delta[i][j] = total
make_heatmap(delta / n, title=f'deltas (averaged), n={n}')
