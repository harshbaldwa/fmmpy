from mayavi import mlab
import numpy as np

def cube_faces(xmin, xmax, ymin, ymax, zmin, zmax):
    faces = []

    x,y = np.mgrid[xmin:xmax:3j,ymin:ymax:3j]
    z = np.ones(y.shape)*zmin
    faces.append((x,y,z))

    x,y = np.mgrid[xmin:xmax:3j,ymin:ymax:3j]
    z = np.ones(y.shape)*zmax
    faces.append((x,y,z))

    x,z = np.mgrid[xmin:xmax:3j,zmin:zmax:3j]
    y = np.ones(z.shape)*ymin
    faces.append((x,y,z))

    x,z = np.mgrid[xmin:xmax:3j,zmin:zmax:3j]
    y = np.ones(z.shape)*ymax
    faces.append((x,y,z))

    y,z = np.mgrid[ymin:ymax:3j,zmin:zmax:3j]
    x = np.ones(z.shape)*xmin
    faces.append((x,y,z))

    y,z = np.mgrid[ymin:ymax:3j,zmin:zmax:3j]
    x = np.ones(z.shape)*xmax
    faces.append((x,y,z))

    return faces

def mlab_plt_cube(xmin,xmax,ymin,ymax,zmin,zmax):
    faces = cube_faces(xmin,xmax,ymin,ymax,zmin,zmax)
    for grid in faces:
        x,y,z = grid
        mlab.mesh(x,y,z,opacity=0.4)

mlab_plt_cube(0,1,0,1,0,1)
mlab.show()