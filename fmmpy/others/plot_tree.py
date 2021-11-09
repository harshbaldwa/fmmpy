from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def draw_node(cx, cy, cz, length, ax, color):
    x = np.linspace(cx-length/2, cx+length/2, 2)
    y = np.linspace(cy-length/2, cy+length/2, 2)
    z = np.linspace(cz-length/2, cz+length/2, 2)
    x1 = np.array([cx-length/2, cx-length/2])
    x2 = np.array([cx+length/2, cx+length/2])
    y1 = np.array([cy-length/2, cy-length/2])
    y2 = np.array([cy+length/2, cy+length/2])
    z1 = np.array([cz-length/2, cz-length/2])
    z2 = np.array([cz+length/2, cz+length/2])

    ax.plot(x1, y1, z, color, linewidth=1+0.5*length)
    ax.plot(x2, y1, z, color, linewidth=1+0.5*length)
    ax.plot(x1, y2, z, color, linewidth=1+0.5*length)
    ax.plot(x2, y2, z, color, linewidth=1+0.5*length)
    ax.plot(x, y1, z1, color, linewidth=1+0.5*length)
    ax.plot(x, y2, z1, color, linewidth=1+0.5*length)
    ax.plot(x, y1, z2, color, linewidth=1+0.5*length)
    ax.plot(x, y2, z2, color, linewidth=1+0.5*length)
    ax.plot(x1, y, z1, color, linewidth=1+0.5*length)
    ax.plot(x2, y, z1, color, linewidth=1+0.5*length)
    ax.plot(x1, y, z2, color, linewidth=1+0.5*length)
    ax.plot(x2, y, z2, color, linewidth=1+0.5*length)


def draw_sphere(cx, cy, cz, r, ax, color):
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    r *= sqrt(3)/2
    x = cx + r*np.cos(u)*np.sin(v)
    y = cy + r*np.sin(u)*np.sin(v)
    z = cz + r*np.cos(v)
    ax.plot_surface(x, y, z, color=color, linewidth=0.5, alpha=0.1)


def plot_tree(cells, cx, cy, cz, length, level, N, x, y, z, spheres, out_r):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = np.ptp([-1, 1])
    ax.set_box_aspect((a, a, a))
    ax.grid(False)

    color = ['b', 'g', 'black', 'orange']

    for i in range(cells):
        if i in spheres:
            draw_sphere(cx[i], cy[i], cz[i], out_r*length/(2**level[i]),
                        ax, 'red')
        # draw_node(cx[i], cy[i], cz[i], length/(2**level[i]), ax, 
        #           color[level[i]])

    ax.scatter3D(x, y, z, c='r', marker='o')
    for i in range(N):
        ax.text(x[i], y[i], z[i], str(i))

    plt.show()
    
if __name__ == "__main__":
    cells = 1
    cx = np.array([0.5])
    cy = np.array([0.5])
    cz = np.array([0.5])
    length = 1  
    level = np.array([0])
    N = 1
    x = np.array([0.5])
    y = np.array([0.5])
    z = np.array([0.5])
    spheres = [0]
    out_r = 1.05
    plot_tree(cells, cx, cy, cz, length, level, N, x, y, z, spheres, out_r)