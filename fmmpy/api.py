import yaml
import pkg_resources


def save_initial_state(N, max_depth, val, x, y, z, out_r, in_r, x_min, y_min,
                       z_min, length, num_p2, dimension, dt, Ns, step, vel_x,
                       vel_y, vel_z):
    """
    Saves the initial state of the simulation.
    """
    val = val.tolist()
    x = x.tolist()
    y = y.tolist()
    z = z.tolist()
    vel_x = vel_x.tolist()
    vel_y = vel_y.tolist()
    vel_z = vel_z.tolist()
    data = {
        "N": N,
        "max_depth": max_depth,
        "part_val": val,
        "part_x": x,
        "part_y": y,
        "part_z": z,
        "vel_x": vel_x,
        "vel_y": vel_y,
        "vel_z": vel_z,
        "out_r": out_r,
        "in_r": in_r,
        "x_min": x_min,
        "y_min": y_min,
        "z_min": z_min,
        "length": length,
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

    PLOT_FILE = pkg_resources.resource_filename('fmmpy', 'data/plot.yaml')
    with open(PLOT_FILE, 'w') as outfile:
        yaml.dump(plot_tree, outfile)
