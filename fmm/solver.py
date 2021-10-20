from compyle.api import get_config
import time
import numpy as np

backend = "cython"
N = 1000
max_depth = 3
direct_call = 1
potential = 0
num_p2 = 24
out_r = 1.1
in_r = 6

# get_config().use_openmp = True

part_val = np.random.random(N)*10 + 1
part_x = np.random.random(N)
part_y = np.random.random(N)
part_z = np.random.random(N)
part_val = part_val.astype(np.float32)
part_x = part_x.astype(np.float32)
part_y = part_y.astype(np.float32)
part_z = part_z.astype(np.float32)

x_min = 0
y_min = 0
z_min = 0
length = 1
dimension = 3

if potential:
    import fmm3 as fmm
    if direct_call:
        res, res_direct = fmm.solver(
            N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, 
            z_min, out_r, in_r, length, num_p2, backend, dimension, 
            direct_call)
        print("Mean Error - ", np.mean(np.abs(res - res_direct)/res_direct),
            "\nMax Error - ", np.max(np.abs(res - res_direct)/res_direct),
            "\nMin Error - ", np.min(np.abs(res - res_direct)/res_direct))
    else:
        start = time.time()
        res = fmm.solver(
            N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, 
            z_min, out_r, in_r, length, num_p2, backend, dimension)
        end = time.time()
        print("Time taken - ", end - start)
else:
    import force2 as fmm
    if direct_call:
        res_x, res_y, res_z, res_dir_x, res_dir_y, res_dir_z = fmm.solver(
            N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, 
            z_min, out_r, in_r, length, num_p2, backend, dimension, 
            direct_call)
        print("Error x: {}\nError y: {}\nError z: {}".format(
            np.mean(np.abs(res_x - res_dir_x)/np.abs(res_dir_x)),
            np.mean(np.abs(res_y - res_dir_y)/np.abs(res_dir_y)),
            np.mean(np.abs(res_z - res_dir_z)/np.abs(res_dir_z))))
    else:
        start = time.time()
        res_x, res_y, res_z = fmm.solver(
            N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, 
            z_min, out_r, in_r, length, num_p2, backend, dimension)
        end = time.time()
        print("Time taken - ", end - start)