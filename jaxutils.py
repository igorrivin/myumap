from jax import grad, jit, vmap, jacfwd
from jax import random
from functools import partial
import jax.numpy as jnp
import jax.numpy as jnp
import jax.random as jr
import numpy as np

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

#f2 = lambda x: jnp.array([ (2 + jnp.cos(2* 2 * jnp.pi * x)) * jnp.cos(3*2 * jnp.pi * x), ( 2 + jnp.cos(2*2 * jnp.pi * x)) * jnp.sin(3*2 * jnp.pi * x), -3 * jnp.sin(2*2 * jnp.pi * x)])

#ffi = np.array(get_irreg_grid(f2, 10000, 0.05))