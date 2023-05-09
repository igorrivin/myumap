import numpy as np
from scipy.stats import ortho_group, special_ortho_group

#generate a flat torus in R4
def rtorus4(a, b, n):
  u = 2 * np.pi * np.random.rand(n)
  v = 2 * np.pi * np.random.rand(n)
  x = a * np.cos(u)
  y = a * np.sin(u)
  z = b * np.cos(v)
  w = b * np.sin(v)
  return np.vstack([x, y, z, w]).T


#generate num random points uniformly distributed in the unit disk in n dimensions

def rand_point_disk(n, num=1):
  prepts = np.random.randn(num, n)
  prenorms = np.linalg.norm(prepts, axis=1).reshape(-1, 1)
  rads = np.sqrt(np.random.rand(num)).reshape(-1, 1)
  pts = prepts * rads/prenorms
  return pts

#return num uniformly distributed random point on the unit sphere in Rn
def rand_point_sph(n, num=1):
  prepts = np.random.randn(num, n)
  prenorms = np.linalg.norm(prepts, axis=1).reshape(-1, 1)
  pts = prepts/prenorms
  return pts

#generate equally spaced points on the unit circle
def do_circ(res, r = 1.0):
  ar = np.arange(0, 2*np.pi, res)
  xs = r * np.cos(ar)
  ys = r * np.sin(ar)
  return np.vstack([xs, ys]).T

#the code below returns uniformly (with respect to the intrinsic metric)
#point on the ellipsoid with the given semi-axis vector a.

class elgen:
  def __init__(self, a):
    themat = np.diag(1/(a * a))
    L = np.linalg.inv(np.linalg.cholesky(themat).T)
    self.L = L

  def __call__(self, ar):
    return (self.L @ ar.T).T

def rand_pt_el(a, n, num=1):
  spts = rand_point_sph(n, num)
  eg = elgen(a)
  return eg(spts)

#the below generates uniform random points on a cylinder
def rcylinder(a, c, n):
    u = 2 * np.pi * np.random.rand(n)
    v = 2 * c * np.random.rand(n)
    x = a * np.cos(u)
    y = a * np.sin(u)
    z = v
    return np.column_stack((x, y, z))

#the code below makes a thickened random circle:

def make_mat(angle):
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

def make_rots(res):
    ar = np.arange(0, 2*np.pi, res)
    return np.array([make_mat(x) for x in ar])

def make_vecs(res, eps, rad=1):
    ar = np.arange(rad-eps, rad+eps, res)
    ar2 = np.zeros(ar.shape)
    return np.vstack([ar, ar2]).T

def make_thick_circ(res1, res2, eps, **kwargs):
    mats = make_rots(res1)
    vecs = make_vecs(res2, eps, **kwargs)
    res = np.einsum('ijk,ik->ij', mats, vecs)
    return res.reshape(-1, 2)