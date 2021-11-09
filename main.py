import argparse

from fmmpy.fmm.hybrid80 import *
from compyle.api import get_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", help="Number of particles in calculations",
                        type=int, default=1000)
    parser.add_argument("-l", "--level", help="Maximum depth of the tree",
                        type=int, default=3)
    parser.add_argument("-p", help="Number of pseudo-particles", type=int,
                        default=6)
    parser.add_argument("-b", "--backend", help="Backend for calculations",
                        type=str, default="cython")
    parser.add_argument("-f", "--force", help="Force calculation",
                        action="store_true")
    parser.add_argument("-omp", "--openmp", help="Use OpenMP for calculations",
                        action="store_true")
    parser.add_argument("-cd", "--compare-direct", help="Compare with direct "
                        "method", action="store_true")

    args = parser.parse_args()
    backends = ["cython", "opencl", "cuda"]
    if args.backend in backends:
        backend = args.backend
    else:
        raise ValueError("Supported backends are '{0}', '{1}' or '{2}'"
                         .format(*backends))

    if args.openmp:
        get_config().use_openmp = True

    np.random.seed(1)

    part_val = np.random.random(args.n).astype(np.float32)
    part_x = np.random.random(args.n).astype(np.float32)
    part_y = np.random.random(args.n).astype(np.float32)
    part_z = np.random.random(args.n).astype(np.float32)

    if args.force:
        res_x, res_y, res_z, res_dir_x, res_dir_y, res_dir_z = solver_force(
            N=args.n, max_depth=args.level, part_val=part_val, part_x=part_x,
            part_y=part_y, part_z=part_z, x_min=0, y_min=0, z_min=0, out_r=1.1,
            in_r=1.05, length=1, num_p2=args.p, backend=backend, dimension=3,
            direct_call=args.compare_direct)
        
        # print(res_x)
        # print(res_dir_x)
        # print()
        # print(res_y)
        # print(res_dir_y)
        # print()
        # print(res_z)
        # print(res_dir_z)
        # print()
        for i in range(args.n):    
            print(abs(res_x[i] - res_dir_x[i]) / abs(res_dir_x[i]))
            print(abs(res_y[i] - res_dir_y[i]) / abs(res_dir_y[i]))
            print(abs(res_z[i] - res_dir_z[i]) / abs(res_dir_z[i]))
        # print("Max Error (x) - ", np.max(np.abs(res_x - res_dir_x) / np.abs(res_dir_x)))
        # print("Max Error (y) - ", np.max(np.abs(res_y - res_dir_y) / np.abs(res_dir_y)))
        # print("Max Error (z) - ", np.max(np.abs(res_z - res_dir_z) / np.abs(res_dir_z)))
        # print()
        # print("Mean Error (x) - ", np.mean(np.abs(res_x - res_dir_x) / np.abs(res_dir_x)))
        # print("Mean Error (y) - ", np.mean(np.abs(res_y - res_dir_y) / np.abs(res_dir_y)))
        # print("Mean Error (z) - ", np.mean(np.abs(res_z - res_dir_z) / np.abs(res_dir_z)))
        
    else:
        res, res_dir = solver(
            N=args.n, max_depth=args.level, part_val=part_val, part_x=part_x,
            part_y=part_y, part_z=part_z, x_min=0, y_min=0, z_min=0, out_r=1.1,
            in_r=1.35, length=1, num_p2=args.p, backend=backend, dimension=3,
            direct_call=args.compare_direct)
        print("Max Error - ", np.max(np.abs(res - res_dir) / res_dir))
        print("Mean Error - ", np.mean(np.abs(res - res_dir) / res_dir))
