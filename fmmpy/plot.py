import numpy as np
from math import sqrt, ceil
from mayavi import mlab
import yaml
import pkg_resources


def draw_node(cx, cy, cz, length, color):
    x = np.linspace(cx - length / 2, cx + length / 2, 3)
    y = np.linspace(cy - length / 2, cy + length / 2, 3)
    z = np.linspace(cz - length / 2, cz + length / 2, 3)

    x_f = np.zeros(48)
    y_f = np.zeros(48)
    z_f = np.zeros(48)

    x_f[0:3] = x
    x_f[3:6] = x[2]
    x_f[6:9] = x[::-1]
    x_f[9:12] = x[0]
    x_f[12:15] = x[0]
    x_f[15:18] = x
    x_f[18:21] = x[2]
    x_f[21:24] = x[::-1]
    x_f[24:27] = x[0]
    x_f[27:30] = x
    x_f[30:33] = x[2]
    x_f[33:36] = x[2]
    x_f[36:39] = x[2]
    x_f[39:42] = x[::-1]
    x_f[42:45] = x[0]
    x_f[45:48] = x[0]

    y_f[0:3] = y[0]
    y_f[3:6] = y[0]
    y_f[6:9] = y[0]
    y_f[9:12] = y[0]
    y_f[12:15] = y
    y_f[15:18] = y[2]
    y_f[18:21] = y[2]
    y_f[21:24] = y[2]
    y_f[24:27] = y[2]
    y_f[27:30] = y[2]
    y_f[30:33] = y[::-1]
    y_f[33:36] = y[0]
    y_f[36:39] = y
    y_f[39:42] = y[2]
    y_f[42:45] = y[::-1]
    y_f[45:48] = y[0]

    z_f[0:3] = z[0]
    z_f[3:6] = z
    z_f[6:9] = z[2]
    z_f[9:12] = z[::-1]
    z_f[12:15] = z[0]
    z_f[15:18] = z[0]
    z_f[18:21] = z
    z_f[21:24] = z[2]
    z_f[24:27] = z[::-1]
    z_f[27:30] = z[0]
    z_f[30:33] = z[0]
    z_f[33:36] = z
    z_f[36:39] = z[2]
    z_f[39:42] = z[2]
    z_f[42:45] = z[2]
    z_f[45:48] = z[::-1]

    radius = length / 100 + 0.0025

    mlab.plot3d(x_f, y_f, z_f, tube_radius=radius, color=color)


def draw_sphere(cx, cy, cz, r, color):
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]
    r *= sqrt(3) / 2
    x = cx + r * np.cos(u) * np.sin(v)
    y = cy + r * np.sin(u) * np.sin(v)
    z = cz + r * np.cos(v)
    mlab.mesh(x, y, z, opacity=0.4, color=(1, 1, 1))


def plot_tree(cells, cx, cy, cz, length, level, N, x, y, z, part2bin, spheres,
              out_r, plot_points, plot_text, save_fig):
    for i in range(cells):
        if i in spheres:
            draw_sphere(cx[i], cy[i], cz[i], out_r *
                        length / (2**level[i]), 'red')
        draw_node(cx[i], cy[i], cz[i], length / (2**level[i]),
                  (0.3 - level[i] * 0.05,
                   0.3 - level[i] * 0.05,
                   0.3 - level[i] * 0.05))

    if plot_points:
        mlab.points3d(x, y, z, np.arange(N), scale_factor=0.025,
                      scale_mode='none', color=(1, 1, 1))

    if plot_text:
        for i in range(N):
            mlab.text3d(x[i], y[i], z[i], str(i) + ", " + str(part2bin[i]),
                        scale=(0.03, 0.03, 0.03))

    if True:
        mlab.points3d(cx, cy, cz, np.arange(cells), scale_factor=0.025,
                      scale_mode='none', color=(0, 0, 0))
        for i in range(cells):
            mlab.text3d(cx[i], cy[i], cz[i], str(i), scale=(0.04, 0.04, 0.04),
                        color=(0, 0, 0))

    if save_fig:
        mlab.savefig("snapshot.obj")
    else:
        mlab.show()


def test_plot(plot_points, plot_text, save_fig):
    N = 1
    cells = 1
    length = 10
    out_r = 1.1
    level = np.zeros(N)
    cx = np.array([5])
    cy = np.array([5])
    cz = np.array([5])
    x = np.array([5])
    y = np.array([5])
    z = np.array([5])
    spheres = [0]
    part2bin = np.zeros(N)

    plot_tree(cells, cx, cy, cz, length, level, N, x, y, z, part2bin, spheres,
              out_r, plot_points, plot_text, save_fig)


def plot(plot_points, plot_text, save_fig):
    PLOT_FILE = pkg_resources.resource_filename('fmmpy', 'data/plot.yaml')
    with open(PLOT_FILE) as file:
        data = yaml.load(file)
        N = data['N']
        cells = data['cells']
        length = data['length']
        out_r = data['out_r']
        level = data['level']
        cx = data['cx']
        cy = data['cy']
        cz = data['cz']
        x = data['part_x']
        y = data['part_y']
        z = data['part_z']
        part2bin = data['part2bin']
        spheres = data['spheres']

    plot_tree(cells, cx, cy, cz, length, level, N, x, y, z, part2bin, spheres,
              out_r, plot_points, plot_text, save_fig)


@mlab.show
@mlab.animate(delay=100, support_movie=True)
def anim(Ns, s, x, y, z):
    for i in range(Ns):
        s.mlab_source.set(x=x[i], y=y[i], z=z[i])
        yield


def find_dist(m, M, x, y, z):
    return sqrt((x[M] - x[m])**2 + (y[M] - y[m])**2 + (z[M] - z[m])**2)


def find_span(x, y, z):
    xM = np.argmax(x)
    xm = np.argmin(x)
    yM = np.argmax(y)
    ym = np.argmin(y)
    zM = np.argmax(z)
    zm = np.argmin(z)
    xl = find_dist(xm, xM, x, y, z)
    yl = find_dist(ym, yM, x, y, z)
    zl = find_dist(zm, zM, x, y, z)
    l_max = max(xl, yl, zl)
    return l_max


def simulate(Ns, N, step):
    p_count = ceil(Ns / step)
    x = np.zeros((p_count, N))
    y = np.zeros((p_count, N))
    z = np.zeros((p_count, N))
    for i in range(p_count):
        infile = pkg_resources.resource_filename(
            'fmmpy', f'data/simulation/sim_{i:02d}.npz')
        npzfile = np.load(infile)
        x[i, :] = npzfile['x']
        y[i, :] = npzfile['y']
        z[i, :] = npzfile['z']
        # print(find_span(x[i, :], y[i, :], z[i, :]))

    mlab.figure(bgcolor=(0, 0, 0), size=(800, 800))
    s = mlab.points3d(x[0], y[0], z[0], scale_factor=25)
    anim(p_count, s, x, y, z)
