import yaml
import pkg_resources
import numpy as np
from compyle.api import annotate, wrap
import compyle.array as ary


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
    part_x, part_y, part_z = wrap(part_x, part_y, part_z, backend=backend)


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


def save_initial_state(N, max_depth, n_max, out_r, in_r, num_p2, dimension,
                       dt=0, Ns=0, step=0):
    """
    Saves the initial state of the simulation.
    """
    data = {
        "N": N,
        "max_depth": max_depth,
        "n_max": n_max,
        "out_r": out_r,
        "in_r": in_r,
        "num_p2": num_p2,
        "dimension": dimension,
        "dt": dt,
        "Ns": Ns,
        "step": step,
    }
    INI_STATE = pkg_resources.resource_filename('fmmpy', 'data/ini_state.yaml')
    with open(INI_STATE, 'w') as outfile:
        yaml.dump(data, outfile)


def read_initial_state():
    """
    Reads the initial state from the file.
    """
    INI_STATE = pkg_resources.resource_filename('fmmpy', 'data/ini_state.yaml')
    T_DESIGN = pkg_resources.resource_filename('fmmpy', 'data/t_design.yaml')

    with open(INI_STATE, 'r') as infile:
        init_state = yaml.load(infile, Loader=yaml.FullLoader)

    with open(T_DESIGN, 'r') as infile:
        data = yaml.load(infile, Loader=yaml.FullLoader)[init_state["num_p2"]]

    return {**init_state, **data}


def visualize(N, cells, length, cx, cy, cz, x, y, z, level, part2bin, out_r):
    cx.pull()
    cy.pull()
    cz.pull()
    x.pull()
    y.pull()
    z.pull()
    level.pull()
    part2bin.pull()
    cx = cx.data[:]
    cy = cy.data[:]
    cz = cz.data[:]
    part_x = x.data[:]
    part_y = y.data[:]
    part_z = z.data[:]
    level = level.data[:]
    part2bin = part2bin.data[:]
    spheres = []

    plot_tree = {"cells": cells, "cx": cx, "cy": cy, "cz": cz,
                 "length": length, "level": level, "N": N, "part_x": part_x,
                 "part_y": part_y, "part_z": part_z, "part2bin": part2bin,
                 "spheres": spheres, "out_r": out_r}

    PLOT_FILE = pkg_resources.resource_filename('fmmpy', 'data/tree.yaml')
    with open(PLOT_FILE, 'w') as outfile:
        yaml.dump(plot_tree, outfile)


def find_span(x, y, z):
    xm = ary.minimum(x)
    xM = ary.maximum(x)
    ym = ary.minimum(y)
    yM = ary.maximum(y)
    zm = ary.minimum(z)
    zM = ary.maximum(z)
    l_max = max(xM - xm, yM - ym, zM - zm)
    l_max *= 1.1
    xm -= l_max / 20
    ym -= l_max / 20
    zm -= l_max / 20
    return l_max, xm, ym, zm
