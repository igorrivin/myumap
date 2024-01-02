from jax import grad, jit, vmap, jacfwd
from jax import random
from functools import partial
import jax.numpy as jnp
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from sympy import symbols, simplify
import sympy as sp
from pyknotid.spacecurves import Knot


def get_uvec2(f):
  tanvec = jit(jacfwd(f))
  return lambda x: tanvec(x)/jnp.linalg.norm(tanvec(x))

def get_cvec(f):
  return get_uvec2(get_uvec2(f))

def get_cvec2(tt):
  return get_uvec2(tt)

def get_frame(f):
  tt = get_uvec2(f)
  tt2 = get_cvec2(tt)
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

def make_rand_knot(decay, l):
  #seed = int(time.time())
  seed = np.random.randint(0, 1e6)
  key = jr.PRNGKey(seed)
  key, subkey = jr.split(key)
  thebatch1x = jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch2x = jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch1y = jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch2y= jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch1z = jr.normal(subkey, shape=(l,))
  key, subkey = jr.split(key)
  thebatch2z= jr.normal(subkey, shape=(l,))
  scaler = jnp.arange(1, l+1, dtype = jnp.float32)**(-decay)
  coefs1x = thebatch1x * scaler
  coefs2x = thebatch2x * scaler
  coefs1y = thebatch1y * scaler
  coefs2y = thebatch2y * scaler
  coefs1z = thebatch1z * scaler
  coefs2z = thebatch2z * scaler
  def make_ser(x):
    c = jnp.arange(1, l+1, dtype = jnp.float32)
    cc = c * 2 * jnp.pi * x
    coses = jnp.cos(cc)
    sines = jnp.sin(cc)
    return jnp.array([jnp.sum(coefs1x * coses) + jnp.sum(coefs2x * sines), jnp.sum(coefs1y * coses) + jnp.sum(coefs2y * sines), jnp.sum(coefs1z * coses) + jnp.sum(coefs2z * sines)])
  return make_ser


""" def make_rand_knot(decay, l):
    seed = jr.PRNGKey(jr.randint(jr.PRNGKey(int(time.time())), shape=(), minval=0, maxval=int(1e6)))
    batches = [jr.normal(jr.split(seed, 2)[0], shape=(l,)) for _ in range(6)]
    
    scaler = jnp.arange(1, l + 1, dtype=jnp.float32) ** (-decay)
    coefs = [batch * scaler for batch in batches]
    
    def make_ser(x):
        c = jnp.arange(1, l + 1, dtype=jnp.float32)
        cc = c * 2 * jnp.pi * x
        coses = jnp.cos(cc)
        sines = jnp.sin(cc)

        cos_terms = jnp.array([jnp.sum(coef * coses) for coef in coefs[:3]])
        sin_terms = jnp.array([jnp.sum(coef * sines) for coef in coefs[3:]])
        return cos_terms + sin_terms

    return make_ser
 """


def get_knot(func, howmany = 1000):
  ff = vmap(func)
  args = jnp.linspace(0, 1, howmany)
  return(ff(args))

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

from fractions import Fraction

def divided_diff(x, y):
    n = len(y)
    coef = [Fraction(yi) for yi in y]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])
    return coef

def newton_interp(coef, x_data, x):
    n = len(x_data) - 1
    result = coef[n]
    for i in range(n - 1, -1, -1):
        result = result * (x - x_data[i]) + coef[i]
    return result

from sympy import symbols, simplify
import sympy as sp

def get_poly(x_data, y_data):
    coef = divided_diff(x_data, y_data)
    x = symbols('x')
    polynomial = coef[0]
    for i in range(1, len(coef)):
        term = coef[i]
        for j in range(i):
            term *= (x - x_data[j])
        polynomial += term
    simplified_poly = simplify(polynomial)
    return simplified_poly


def get_alex_poly(func, numpts = 100):
   foo = get_knot(func)
   frog = Knot(foo)
   apoly = frog.alexander_polynomial()
   gcode = frog.gauss_code()
   if len(gcode) < 3:
      return 1
   x = list(range(1, numpts+1))
   y = [frog.alexander_polynomial(variable = xx) for xx in x]
   return get_poly(x, y)

""" 
# Example usage
x_data = [0, 1, 2, 5]
y_data = [1, 2, 1, 10]
coef = divided_diff(x_data, y_data)

# Interpolate
x = 3
print(newton_interp(coef, x_data, x))
 """
#f2 = lambda x: jnp.array([ (2 + jnp.cos(2* 2 * jnp.pi * x)) * jnp.cos(3*2 * jnp.pi * x), ( 2 + jnp.cos(2*2 * jnp.pi * x)) * jnp.sin(3*2 * jnp.pi * x), -3 * jnp.sin(2*2 * jnp.pi * x)])

#ffi = np.array(get_irreg_grid(f2, 10000, 0.05))