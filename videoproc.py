import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import faiss
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors as nn


import math

import sys

def get_frame_list(fname, freq = None):
    #freq tells us how many times a second we sample. None tells us that we sample every frame.
    cap = cv2.VideoCapture(fname)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if freq is None:
        frame_step = 1
    else:
        frame_step = math.floor(frame_rate /freq)
    if not cap.isOpened():
        raise Exception(f'Error opening video file {fname}')
    flist = []
    count = 0
    while True:
        ret, frame = cap.read()
    
        if not ret:
            break  # Video file ended

        frame = np.ndarray.flatten(frame)

        if count % frame_step == 0:  # If this frame number is a multiple of the frame step
        #cv2.imwrite('frame{:d}.jpg'.format(count), frame)
            flist.append(frame)
        print(count, len(flist))

        count += 1
    cap.release()
    tmpdata = np.vstack(flist)
    data = np.ascontiguousarray(tmpdata.astype(np.float32))
    return data

def do_nn(data, k=2, algo='faiss'):
    print(algo)
    if algo == 'faiss':
        d = data.shape[1]
        index = faiss.IndexFlatL2(d)   # build the index
        index.add(data)                  # add vectors to the index
        D, I = index.search(data, k+1)
        D = D[:, 1:]
        I = I[:, 1:]
        return D, I
    neigh = nn()
    neigh.fit(data)
    s=neigh.kneighbors(n_neighbors=2, return_distance=True)
    return s



def calculate_intrinsic_dimension(distances):
    # calculate the ratio of distances
    ratios = distances[:, 0] / distances[:, 1]

    # ignore ratios where distance to first neighbour is zero
    ratios = ratios[~np.isinf(ratios)]

    # calculate the intrinsic dimension
    intrinsic_dim = len(ratios) / -np.sum(np.log(ratios))

    return intrinsic_dim, ratios

def plot_histogram(ratios, filename=None):
    # plot the histogram of ratios
    plt.hist(np.power(ratios, 1), bins=20)
    if filename:
        plt.savefig(filename)
    plt.show()

#from sklearn.neighbors import NearestNeighbors as nn
#neigh=nn()
#neigh.fit(Idata)
#s=neigh.kneighbors(n_neighbors=2, return_distance=True)

def main(args):
    fname = args.file
    freq = args.freq
    outfile = args.outfile
    data = get_frame_list(fname, freq)
    algo = args.algo
    print(data.shape)
    s = do_nn(data, k=2, algo=algo)
    print("Found neighbors")
    intrinsic_dim, ratios = calculate_intrinsic_dimension(s[0])
    print('Intrinsic Dimension =', intrinsic_dim)
    print('Number of coinciding images=', np.sum(np.isinf(s[0][:, 0])))
    print('Dataset cardinality=', len(ratios))
    plot_histogram(ratios, outfile)
    return 0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file', help = 'where the movie lives')
    parser.add_argument('--freq', type = int)
    parser.add_argument('--outfile', help='where should we output the histogram')
    parser.add_argument('--algo', help = 'what algorithm to use')
    args = parser.parse_args()
    sys.exit(main(args))
