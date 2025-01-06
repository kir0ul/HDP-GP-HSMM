# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from HDPGPHSMM_segmentation import GPSegmentation
import time
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from pathlib import Path
import shutil


def conv_time2ptsnb(time_min, time_max, freq):
    dt = 1/freq
    max_len = int(np.round(time_max/dt))
    min_len = int(np.round(time_min/dt))
    avg_len = int(np.round(np.mean([max_len, min_len])))
    skip_len = int(np.round(min_len/2))
    print(f"MAX_LEN: {max_len}")
    print(f"MIN_LEN: {min_len}")
    print(f"AVG_LEN: {avg_len}")
    print(f"SKIP_LEN: {skip_len}")
    return max_len, min_len, avg_len, skip_len

maxlen, minlen, avelen, skiplen = conv_time2ptsnb(
    time_min=2,
    time_max=7,
    freq=17
)

data_path = Path(".") / "Input_Data" / "LASADataset"

files =  list(data_path.glob("Bended*"))
data_dimensions = None
for fname in files:
    print(fname.absolute())
    data = np.loadtxt(fname)
    if data_dimensions is None:
        data_dimensions = data.shape
    if data.shape != data_dimensions:
        raise ValueError("All the data files don't have the same dimensions")
dim = data.shape[1]
print(f"DIM = {dim}")

learn_path = data_path / "learn"
recog_path = data_path / "recog"
if learn_path.exists():
    shutil.rmtree(learn_path)
if recog_path.exists():
    shutil.rmtree(recog_path)


def learn( savedir, dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen ):
    gpsegm = GPSegmentation( dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen)

    # files =  [ "Input_Data/testdata2dim_%03d.txt" % j for j in range(4) ]
    gpsegm.load_data( files )
    liks = []

    start = time.time()
    #iteration (default: 10)
    for it in range( 10 ):
        print( "-----", it, "-----" )
        gpsegm.learn()
        numclass = gpsegm.save_model( savedir )
        print( "lik =", gpsegm.calc_lik() )
        liks.append(gpsegm.calc_lik())
    print ("liks: ",liks)
    print( time.time()-start )

    #plot liks
    plt.clf()
    plt.plot( range(len(liks)), liks )
    print(f"Saved: {os.path.join( savedir,'liks.png')}")
    plt.savefig( os.path.join( savedir,"liks.png") )

    return numclass


def recog( modeldir, savedir, dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen ):
    print ("class", initial_class)
    gpsegm = GPSegmentation( dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen)

    # gpsegm.load_data( [ "Input_Data/testdata2dim_%03d.txt" % j for j in range(4) ] )
    gpsegm.load_data(files)
    gpsegm.load_model( modeldir )


    start = time.time()
    gpsegm.recog()
    print( "lik =", gpsegm.calc_lik() )
    print( time.time()-start )
    gpsegm.save_model( savedir )


def main():
    #parameters
    # dim = 2
    gamma = 2.0
    eta = 5.0

    initial_class = 1

    # avelen = 15
    # maxlen = int(avelen + avelen*0.25)
    # minlen = int(avelen*0.25)
    print(maxlen, minlen)
    # skiplen = 1

    #learn
    print ( "=====", "learn", "=====" )
    # recog_initial_class = learn( "learn/", dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen )
    recog_initial_class = learn( learn_path, dim, gamma, eta, initial_class, avelen, maxlen, minlen, skiplen )
    #recognition
    print ( "=====", "recognition", "=====" )
    # recog( "learn/", "recog/", dim, gamma, eta, recog_initial_class, avelen, maxlen, minlen, skiplen )
    recog( learn_path, recog_path, dim, gamma, eta, recog_initial_class, avelen, maxlen, minlen, skiplen )
    return

if __name__=="__main__":
    main()
