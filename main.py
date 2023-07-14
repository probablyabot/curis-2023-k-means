from sdp import *
from time import time
import argparse
from matplotlib import pyplot as plt


def get_data(args):
    if args.manual:
        data = np.array([parse_point(f'Data point #{i}: ', args.dimension)
                         for i in range(args.pts_per_cluster)])
    else:
        centers = [parse_point(f'Center #{i}: ', args.dimension)
                   for i in range(args.num_clusters)]
        if args.polygon:
            data = gen_polygon_clusters(args.num_clusters, args.pts_per_cluster,
                                        args.radius, centers)
        else:
            data = gen_clusters(args.num_clusters, args.pts_per_cluster,
                                args.dimension, args.radius, centers)
    data = np.array([[-0.601,  0.098],
             [-0.373, -0.254],
             [ 0.601, -0.011],
             [ 0.355, -0.457],
             [ 0.525,  0.704],
             [-0.526,  0.346],
             [ 0.184,  0.69 ],
             [ 0.436, -0.704],
             [-0.179,  0.2  ],
             [-0.111, -0.039]])
    data = np.vstack([data, (10,10) + data])
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Three possible use cases:
    # 1. Sampling from balls: -nc 2 -ppc 10 -k 2 [-v] [-sep] [-ng]  (2 clusters of 10 points, k=2)
    # 2. Regular polygons: -p -nc 2 -ppc 5 -k 2 [-v] [-sep] [-ng]   (2 pentagons, k=2)
    # 3. Manual data entry: -m -ppc 20 -k 2 [-v] [-sep] [-ng]       (20 data points, k=2)
    # For use cases 1 and 2, centers are given via user input.
    parser.add_argument('-m', '--manual', action='store_true')
    parser.add_argument('-p', '--polygon', action='store_true')
    parser.add_argument('-nc', '--num_clusters', type=int)
    parser.add_argument('-ppc', '--pts_per_cluster', type=int)
    parser.add_argument('-k', type=int)
    parser.add_argument('-d', '--dimension', type=int, default=2)
    parser.add_argument('-r', '--radius', type=float, default=1.0)
    parser.add_argument('-ng', '--no_gurobi', action='store_true')
    parser.add_argument('-sep', '--separate_opt', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    data = get_data(args)
    k = args.k
    if args.verbose:
        print('Input data:\n', np.around(data, 3))
        # TODO: create separate plotting function, highlight clusters
        plt.scatter(data[:, 0], data[:, 1])
        plt.show()

    start_t = time()
    m, cost = sdp_k_means(data, k)
    sdp_t = time()
    if not args.no_gurobi:
        if args.separate_opt:
            opt = optimal_separate(data, k, args.pts_per_cluster)
        else:
            opt = optimal_k_means(data, k)
        opt_t = time()
        print('Optimal objective function value:', round(opt, 3))
        print(f'Gurobi running time: {opt_t - sdp_t} seconds')
    print('SDP objective function value:', round(cost, 3))
    if args.verbose:
        print('SDP solver returned matrix:\n', np.around(m, 3))
        print('Trace:\n', np.around(np.trace(m), 3))
    print(f'SDP running time: {sdp_t - start_t} seconds')
