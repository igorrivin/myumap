import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, SpectralEmbedding
import umap
import trimap
import pacmap


def umap_compare(data,*, alg = 'umap',indim = 3, outdim = 2, **kwargs):
  if alg == 'umap':
    reducer = umap.UMAP(n_components = outdim, **kwargs)
  elif alg == 'trimap':
    reducer = trimap.TRIMAP(n_dims = outdim, **kwargs )
  elif alg == 'pacmap':
    reducer = pacmap.PaCMAP(n_components = outdim, **kwargs)
  elif alg =='tsne':
    reducer = TSNE(n_components = outdim, **kwargs)
  elif alg == 'spec':
    reducer =  SpectralEmbedding(n_components = outdim, **kwargs)
  else:
    return
  #data = rand_pt(indim, n)
  reduced = reducer.fit_transform(data)
  f = plt.figure()
  f.set_figwidth(15)
  f.set_figheight(30)
  if indim == 3:
    ax1 = f.add_subplot(211,projection='3d')
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax1.scatter(xs, ys, zs)
    ax1.set_title("original data")
  elif indim == 2:
    ax1 = f.add_subplot(211)
    xs = data[:, 0]
    ys = data[:, 1]
    ax1.scatter(xs, ys)
    ax1.set_title("original data")
  else:
    pass
  if outdim == 3:
    ax2 = f.add_subplot(212, projection='3d')
    xs = reduced[:, 0]
    ys = reduced[:, 1]
    zs = reduced[:, 2]
    ax2.scatter(xs, ys, zs, s=1)
    ax2.set_title("reduced data")
  else:
    ax2 = f.add_subplot(212)
    xs = reduced[:, 0]
    ys = reduced[:, 1]
    ax2.scatter(xs, ys, s=1)
    ax2.set_title("reduced data")
  plt.show()
