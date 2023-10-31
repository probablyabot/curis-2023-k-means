import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


def gen_polygon(n, r=1, cx=0, cy=0):
    t = 2 * np.pi / n
    pts = [[r*np.cos(t*i), r*np.sin(t*i)] for i in range(n)]
    return (cx, cy) + np.array(pts)


def ngon_dual(n):
    pts = gen_polygon(n, n / (2 * np.pi))

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


# takes in an n x n numpy array and creates a heatmap
def make_heatmap(arr, title='Heatmap'):
    plt.imshow(arr, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xticks(np.arange(0, n, 2))
    plt.yticks(np.arange(0, n, 2))
    plt.gca().invert_yaxis()
    plt.show()


n = 20
a, b, d = ngon_dual(n)
print('alpha:', np.round(a, 3))
make_heatmap(b, f'betas, n={n}')
delta = np.zeros((n, n))
for i in range(1, n - 1):
    for j in range(1, n - i):
        total = 0
        for k in range(n):
            k1, k2 = sorted([(k+i) % n, (k-j) % n])
            total += d[k][k1][k2]
        delta[i][j] = total
make_heatmap(delta / n, f'deltas (averaged), n={n}')
