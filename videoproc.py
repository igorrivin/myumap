import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import faiss
from argparse import ArgumentParser
from sklearn.neighbors import NearestNeighbors as nn
import pywt
from sklearn.cluster import SpectralClustering



import math

import sys

def do_one_color(frame,keep, n=4, w = 'haar'):
    coeffs = pywt.wavedec2(frame,wavelet=w,level=n)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    Csort = np.sort(np.abs(coeff_arr.reshape(-1)))
    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind # Threshold small indices
    coeffs_filt = pywt.array_to_coeffs(Cfilt,coeff_slices,output_format='wavedec2')
    recon = pywt.waverec2(coeffs_filt,wavelet=w)
    return recon


def transf(frame, wavelet):
    if wavelet is None:
        return frame
    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]
    frame[:, :, 0] = do_one_color(blue, wavelet)
    frame[:, :, 1] = do_one_color(green, wavelet)
    frame[:, :, 2] = do_one_color(red, wavelet)
    return frame


def get_frame_list(fname, freq=None, max_count=1000, get_chan = None, wavelet = None, get_orig = False, start = 0:
    chandict = {'B':0, 'G':1, 'R':2}
    cap = cv2.VideoCapture(fname)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_step = 1 if freq is None else math.floor(frame_rate / freq)
    orig_list = []

    if not cap.isOpened():
        raise Exception(f'Error opening video file {fname}')

    # Read the first frame to get its size
    ret, frame = cap.read()
    if not ret:
        raise Exception('Could not read first frame')

    # Preallocate a numpy array to hold all the frames
    frame = transf(frame, wavelet)
    if get_orig:
        orig_list.append(frame)
    frame = np.ndarray.flatten(frame)
    l = len(frame)
    if get_chan is not None:
        l1 = l//3
    else:
        l1 = l
    flist = np.empty((max_count, l1), dtype=np.float32)

    count = 0
    while count < max_count + start:
        if count < start:
            count += 1
            continue
        if count % frame_step == 0:  # If this frame number is a multiple of the frame step
            newframe = np.ndarray.flatten(frame)
            if get_chan is None:
                flist[count] = newframe
            else:
                flist[count] = newframe[np.arange(chandict[get_chan], l, 3)]

        count += 1

        # Read next frame for the next iteration
        ret, frame = cap.read()
        if not ret:
            break  # Video file ended
        frame = transf(frame, wavelet)
        if get_orig:
            orig_list.append(frame)

    cap.release()
    # No need for ascontiguousarray or astype, flist is already a contiguous float32 array
    return flist[:count], orig_list  # Return the part of the array that we filled and the list of original frames

""" def get_frame_list(fname, freq = None, max_count = 1000):
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
    while count < max_count:
        ret, frame = cap.read()
    
        if not ret:
            break  # Video file ended

        frame = np.ndarray.flatten(frame)

        if count % frame_step == 0:  # If this frame number is a multiple of the frame step
        #cv2.imwrite('frame{:d}.jpg'.format(count), frame)
            flist.append(frame)
        #print(count, len(flist))

        count += 1
    cap.release()
    tmpdata = np.vstack(flist)
    data = np.ascontiguousarray(tmpdata.astype(np.float32))
    return data """

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
    ratios = ratios[~np.isinf(ratios) & ~np.isnan(ratios) & (ratios > 0)]

    # calculate the intrinsic dimension
    intrinsic_dim = len(ratios) / -np.sum(np.log(ratios))
    print('Intrinsic Dimension =', intrinsic_dim)
    print('Dataset cardinality=', len(ratios))
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

def get_dim(data, algo = None, do_plot = False, filename = None):
   s = do_nn(data, algo = algo)
   print('Number of coinciding images=', np.sum(np.isinf(s[0][:, 0])))
   d, r = calculate_intrinsic_dimension(s[0])
   if do_plot:
       plot_histogram(r, filename)
   return calculate_intrinsic_dimension(s[0])

def do_data(fname,  freq, maxcount, outfile=None, algo=None, get_chan = None, wavelet = None):
    data, _ = get_frame_list(fname, freq, max_count = maxcount, get_chan = get_chan, wavelet=wavelet)
    print(data.shape)
    intrinsic_dim, ratios = get_dim(data, algo=algo, do_plot = True, filename = outfile)
    # s = do_nn(data, k=2, algo=algo)
    # print("Found neighbors")
    # intrinsic_dim, ratios = calculate_intrinsic_dimension(s[0])
    print('Intrinsic Dimension =', intrinsic_dim)
    #print('Number of coinciding images=', np.sum(np.isinf(s[0][:, 0])))
    print('Dataset cardinality=', len(ratios))
    #plot_histogram(ratios, outfile)

def main(args):
    fname = args.file
    freq = args.freq
    outfile = args.outfile
    maxcount = args.maxcount
    algo = args.algo
    getchan = args.get_chan
    wavelet = args.wavelet
    do_data(fname, freq, maxcount, outfile, algo, get_chan = getchan, wavelet = wavelet)
    """ data = get_frame_list(fname, freq)
    algo = args.algo
    print(data.shape)
    s = do_nn(data, k=2, algo=algo)
    print("Found neighbors")
    intrinsic_dim, ratios = calculate_intrinsic_dimension(s[0])
    print('Intrinsic Dimension =', intrinsic_dim)
    print('Number of coinciding images=', np.sum(np.isinf(s[0][:, 0])))
    print('Dataset cardinality=', len(ratios))
    plot_histogram(ratios, outfile) """
    return 0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('file', help = 'where the movie lives')
    parser.add_argument('--freq', type = int)
    parser.add_argument('--outfile', help='where should we output the histogram')
    parser.add_argument('--algo', help = 'what algorithm to use')
    parser.add_argument('--maxcount', type = int, default = 1000, help='maximum number of frames to process')
    parser.add_argument('--get_chan', help="should we just get one channel?")
    parser.add_argument('--wavelet', type = float, help="if present, tells us what percentage of wavelet coefficients to keep")
    args = parser.parse_args()
    sys.exit(main(args))
