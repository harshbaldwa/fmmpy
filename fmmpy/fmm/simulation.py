import pkg_resources
import numpy as np
from compyle.api import annotate, wrap


def save_sim(part_x, part_y, part_z, sim, backend):
    part_x.pull()
    part_y.pull()
    part_z.pull()
    part_x = part_x.data[:]
    part_y = part_y.data[:]
    part_z = part_z.data[:]
    outfile = pkg_resources.resource_filename(
        'fmmpy', f'data/simulation/sim_{sim:02d}.npz')
    np.savez(outfile, x=part_x, y=part_y, z=part_z)
    part_x, part_y, part_z = wrap(
        part_x, part_y, part_z, backend=backend)


@annotate(i="int", dt="float", gfloatp="part_x, part_y, part_z, vel_x, vel_y, "
          "vel_z, res_x, res_y, res_z")
def timestep(i, part_x, part_y, part_z, vel_x, vel_y, vel_z, res_x, res_y,
             res_z, dt):
    vel_x[i] = 0.9983 * vel_x[i] + res_x[i] * dt
    vel_y[i] = 0.9983 * vel_y[i] + res_y[i] * dt
    vel_z[i] = 0.9983 * vel_z[i] + res_z[i] * dt

    part_x[i] += vel_x[i] * dt
    part_y[i] += vel_y[i] * dt
    part_z[i] += vel_z[i] * dt
