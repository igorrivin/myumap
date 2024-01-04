import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from jaxgeom import *


def make_tube(f2, numpts = 10000, rad=0.1, alpha = 0.01, colab = True):

  ffi = np.array(get_irreg_grid(f2, numpts, rad))
  # Example point cloud data
  points = ffi

  # Create a PolyData object
  cloud = pv.PolyData(points)

  # Attempt to create a surface via Delaunay 3D
  surf = cloud.delaunay_3d(alpha=alpha)

  # Extract the surface
  surface = surf.extract_surface()



  # Assume 'surface' is your PyVista surface object
  vertices = surface.points
  faces = surface.faces.reshape((-1, 4))[:, 1:4]

  if colab:
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting each triangle in the surface
    for face in faces:
        triangle = vertices[face]
        poly = Poly3DCollection([triangle], alpha=0.01, edgecolor='k')
        ax.add_collection3d(poly)

    # Auto scaling the axes
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])

    # Show the plot
    plt.show()
  else:
    surface.plot(opacity = 0.1)
  return ffi

