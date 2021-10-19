import time
import numpy as np
import fmm as fmm

backend = "opencl"
N = 5000000
max_depth = 7
direct_call = 0
num_p2 = 12
out_r = 1.1
in_r = 1.05

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

if direct_call:
    res, res_direct = fmm.solver(
        N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min,
        out_r, in_r, length, num_p2, backend, dimension, direct_call)
    print("Mean Error - ", np.mean(np.abs(res - res_direct)/res_direct),
        "\nMax Error - ", np.max(np.abs(res - res_direct)/res_direct))
else:
    start = time.time()
    res = fmm.solver(
        N, max_depth, part_val, part_x, part_y, part_z, x_min, y_min, z_min, 
        out_r, in_r, length, num_p2, backend, dimension)
    end = time.time()
    print("Time taken - ", end - start)
