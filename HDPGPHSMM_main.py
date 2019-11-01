# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from HDPGPHSMM_segmentaion import GPSegmentation
import time
import matplotlib.pyplot as plt
import os

def learn( savedir, dim, gamma, eta ):
    gpsegm = GPSegmentation( dim, gamma, eta) 

    files =  [ "testdata2dim_%03d.txt" % j for j in range(5) ]
    gpsegm.load_data( files )
    liks = []
    
    start = time.clock()
    #iteration (default: 10)
    for it in range(10):
        print( "-----", it, "-----" )
        gpsegm.learn()
        gpsegm.save_model( savedir )
        print( "lik =", gpsegm.calc_lik() )
        liks.append(gpsegm.calc_lik())
    print ("liks: ",liks)
    print( time.clock()-start )
    
    #plot liks
    plt.clf()
    plt.plot( range(len(liks)), liks )
    plt.savefig( os.path.join( savedir,"liks.png") )
        
    return gpsegm.calc_lik()


def recog( modeldir, savedir, dim, gamma, eta ):
    gpsegm = GPSegmentation( dim, gamma, eta)

    gpsegm.load_data( [ "testdata2dim_%03d.txt" % j for j in range(4) ] )
    gpsegm.load_model( modeldir )


    start = time.clock()
    for it in range(5):
        print( "-----", it, "-----" )
        gpsegm.recog()
        print( "lik =", gpsegm.calc_lik() )
    print( time.clock()-start )
    gpsegm.save_model( savedir )



def main():
    #parameters
    dim = 2
    gamma = 1.0
    eta = 10.0
    
    #learn
    learn( "learn/", dim, gamma, eta )
    #recognition
    recog( "learn/", "recog/", dim, gamma, eta )
    return

if __name__=="__main__":
    main()