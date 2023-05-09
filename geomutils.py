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

#more random circles:

def make_thick_rand_circ2(num, eps, gap=0):
  res = []
  for i in range(num):
    mats = make_rots(1,gap=gap)
    vecs = make_vecs(1, eps)
    tmp = mats[0] @ vecs[0]
    res.append(tmp)
  return np.vstack(res)

#random and non-random intervals

def make_rand_int(num, len):
  tmp = len * np.random.rand(num)
  tmp2 = np.zeros(num)
  return np.vstack([tmp, tmp2]).T

def make_reg_int(num, len):
  tmp = np.linspace(start=0, stop=len, num = num)
  tmp2 = np.zeros(num)
  return np.vstack([tmp, tmp2]).T

#now we generate random points on tubes around curves:

from jax import grad, jit, vmap, jacfwd
from jax import random
from functools import partial
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

def get_uvec2(f):
  tanvec = jit(jacfwd(f))
  return lambda x: tanvec(x)/jnp.linalg.norm(tanvec(x))

def get_cvec(f):
  return get_uvec2(get_uvec2(f))

def get_frame(f):
  tt = get_uvec2(f)
  tt2 = get_cvec(f)
  def first2(t):
    x = tt(t)
    y = tt2(t)
    tt3 = (jnp.cross(x, y))
    return jnp.array([x, y, tt3])
  return jit(first2)

def get_point(frame, s):
  v1 = frame[1, :]
  v2 = frame[2, :]
  return jnp.cos(s) * v1 + jnp.sin(s) * v2

def get_grid(f, eps):
  ffunc = get_frame(f)
  def grid(t, s):
    base = f(t)
    frame = ffunc(t)
    return base + eps * get_point(frame, s)
  return grid

def get_grida(f, eps):
  ffunc = get_frame(f)
  def grid(ar):
    t = ar[0]
    s = ar[1]
    base = f(t)
    frame = ffunc(t)
    return base + eps * get_point(frame, s)
  return grid

def get_gridb(f, eps):
  ffunc = get_frame(f)
  def grid(ar):
    t = ar[0]
    s = ar[1]
    r = jnp.sqrt(ar[2])
    base = f(t)
    frame = ffunc(t)
    return base + eps * r * get_point(frame, s)
  return grid

@partial(jit, static_argnames=['f', 'num1', 'num2'])
def get_reg_grid(f, num1, num2, eps):
  tarray = jnp.linspace(start = 0.0, stop = 1.0, num = num1)
  sarray = jnp.linspace(start = 0.0, stop = 2 * jnp.pi, num = num2)
  g = get_grid(f, eps)
  g = vmap(g, in_axes=(None, 0))
  g = vmap(g, in_axes=(0, None))
  return jnp.vstack(g(tarray, sarray))


@partial(jit, static_argnames=['f', 'num'])
def get_irreg_grid(f, num, eps):
  parray = jnp.array(np.random.rand(num, 2) * np.array([1, 2 * np.pi]))
  g = get_grida(f, eps)
  g = vmap(g, in_axes=( 0))
  return jnp.vstack(g(parray))

@partial(jit, static_argnames=['f', 'num'])
def get_irreg_grid_full(f, num, eps):
  parray = jnp.array(np.random.rand(num, 3) * np.array([1, 2 * np.pi, 1]))
  g = get_gridb(f, eps)
  g = vmap(g, in_axes=( 0))
  return jnp.vstack(g(parray))

#here is an example of use:

#f2 = lambda x: jnp.array([ (2 + jnp.cos(2* 2 * jnp.pi * x)) * jnp.cos(3*2 * jnp.pi * x), ( 2 + jnp.cos(2*2 * jnp.pi * x)) * jnp.sin(3*2 * jnp.pi * x), -3 * jnp.sin(2*2 * jnp.pi * x)])

#ffi = np.array(get_irreg_grid(f2, 10000, 0.05))

