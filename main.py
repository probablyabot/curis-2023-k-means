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
        elif args.simplex:
            data = gen_simplex_clusters(args.num_clusters, args.pts_per_cluster,
                                        centers)
        else:
            data = gen_clusters(args.num_clusters, args.pts_per_cluster,
                                args.dimension, args.radius, centers)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Three use cases:
    # 1. Sampling from balls: -nc 2 -ppc 10 -k 2 [-v] [-ng]  (2 clusters of 10 points, k=2)
    # 2. Regular polygons: -p -nc 2 -ppc 5 -k 2 [-v] [-ng]   (2 pentagons, k=2)
    # 3. Manual data entry: -m -ppc 20 -k 2 [-v] [-ng]       (20 data points, k=2)
    # For use cases 1 and 2, centers are given via user input.
    parser.add_argument('-m', '--manual', action='store_true')
    parser.add_argument('-p', '--polygon', action='store_true')
    parser.add_argument('-s', '--simplex', action='store_true')
    parser.add_argument('-nc', '--num_clusters', type=int, default=1)
    parser.add_argument('-ppc', '--pts_per_cluster', type=int, default=3)
    parser.add_argument('-k', type=int)
    parser.add_argument('-d', '--dimension', type=int, default=2)
    parser.add_argument('-r', '--radius', type=float, default=1.0)
    parser.add_argument('-ng', '--no_gurobi', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-lp', action='store_true')
    parser.add_argument('-tri', action='store_true')

    args = parser.parse_args()
    data = get_data(args)
    # data = gen_clique_embeddings(6, 4)
    k = args.k
    if args.verbose:
        print('Input data:\n', np.around(data, 3))
        # TODO: create separate plotting function, highlight clusters and centers
        plt.scatter(data[:, 0], data[:, 1])
        plt.axis('equal')
        plt.show()

    start_t = time()
    print('Running LP solver...')
    m, cost, duals = sdp_k_means(data, k, not args.lp, args.tri)
    sdp_t = time()
    print('LP solver finished.')
    if not args.no_gurobi:
        print('Finding integer solution...')
        if args.polygon:
            opt = optimal_polygon(data, args.pts_per_cluster, k)
        elif args.simplex:
            opt = optimal_simplex(data, args.pts_per_cluster, k)
        else:
            opt = optimal_k_means(data, k)
        opt_t = time()
        print('Integer solution found.')
        print('Integer solution cost:', round(opt, 3))
        print(f'Integer solution running time: {round(opt_t - sdp_t, 3)} seconds')
        print(f'Ratio:', round(opt / cost, 3))
    print('SDP solution cost:', round(cost, 3))
    if args.verbose:
        print('SDP solver returned matrix:\n', np.around(m, 3))
        print('Trace:\n', np.around(np.trace(m), 3))
        # for d in duals:
        #     print('Dual variables:', np.around(d, 3))
    print(f'SDP running time: {round(sdp_t - start_t, 3)} seconds')
