#! /usr/bin/env /usr/bin/python3

#  cluster_weather_labels.py
# 
#  Copyright (C) 2020-2022 Scott L. Williams.
# 
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#

cluster_labels_copyright = 'cluster_weather_labels.py Copyright (c) 2020-2023 Scott L. Williams, released under GNU GPL V3.0'

# poli batch implementation to train SOM and writes out labels
# this version of SOM training is used to cluster pixels with similar labels

import os
import sys
import getopt

import numpy as np

from npy_source_pack import npy_source
from msom_pack import msom
from render_pack import render

nclasses = 25                       # must match classes in SOM map (eg. *.labels)

# instantiate the operators

# numpy source
src = npy_source.npy_source( 'npy_source' )

# mini-som
ms = msom.msom( 'msom' )
ms.params.shape = (3,4)             # net grid shape (3,4) 
ms.params.sigma = 2
ms.params.nepochs = 30              # determines how many samples
                                    # should be considered. an epoch
                                    # is a full samping of the data.
                                    # multiples epochs allow for pixel
                                    # reconsideration and node reassigment.
                                    
ms.params.thresh = None             # set to stop processing below this value
                                    # set low to see QE values per epoch
                                    # otherwise set to None

ms.params.rate = 0.3                # 0.5 for jan-mar_2019 for decay 3
ms.params.init_weights = 'pca'
ms.params.neighborhood_function = 'gaussian'
ms.params.topology = 'hexagonal'
ms.params.activation_distance = 'euclidean'
ms.params.output_type = 'labels'
ms.params.apply_classification = True
ms.params.seed = None
ms.params.rorder = True
ms.params.order_by_size = False
ms.params.decay_function = 4        # linear decay function; was 3 for jan-mar_2019 with nepochs=50
ms.params.calc_epoch_QE = True

# render labels as image
rndr = render.render( 'render' )
rndr.readlut( './luts/sixteenthbow.lut' )

# -----------------------------------------------------------

def cluster( infile, outdir ) :

   print( 'output directory for clustering:', outdir,
          file=sys.stderr, flush=True )
   
   # read the daily labels file as training data
   print( 'reading datafile: ' + infile + '...',
          file=sys.stderr, flush=True, end='' )
   src.params.filepath = infile
   src.run()
   print( 'done', file=sys.stderr, flush=True )

   print( 'number of epochs=', ms.params.nepochs, file=sys.stderr, flush=True )
   print( 'learning rate=', ms.params.rate, file=sys.stderr, flush=True )

   # for each pixel make a label histogram, use this as a feature vector
   print( 'making histograms...', file=sys.stderr, flush=True, end='' )
   height, width, nbins = src.sink.shape
   hist = np.empty( (height,width,nclasses), dtype=np.int64 )

   # construct an image of histograms and cluster these
   for j in range( height ):
       for i in range( width ):

          # for each pixel make a histogram of labels
          h,b = np.histogram( src.sink[j,i,:], bins=nclasses,
                              range=(0.0,float(nclasses-1)) )
          hist[j,i,:] = h

          # NOTE: if you get this error:
          #       minisom.py:486: RuntimeWarning: invalid value encountered in sqrt
          #       return sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)
          # it is saying that the data type cannot fit the result a square operation.
          # eg, dtype=uint16 but 364*364=132496 which is bigger than 2^16 - 1 = 65535


   print( 'done', file=sys.stderr, flush=True )

   '''                                                                                        
   # report max and min
   mx = np.amax( hist )
   mn = np.amin( hist )
   print( mx, mn, file=sys.stderr, flush=True )
   '''                                                                                        
   
   # rational for not normalizing: feature vector components
   # have same units, ie. number of days

   # make numpy file for graphics
   hist.dump( outdir + '/hist.npy' )

   # link output to msom input and run
   print( 'training ...', file=sys.stderr, flush=True )

   ms.params.mapfile_prefix = outdir + '/cluster'
   ms.source = hist
   ms.run() # train

   # save the cluster labels
   ms.sink.dump( outdir + '/cluster.npy' )

   # render as image
   rndr.params.filepath = outdir + '/cluster.jpg'
   rndr.source = ms.sink
   rndr.run()

# ---------------------------------------------------------------


def usage( self ):
        print( 'usage: cluster_labels.py', file=sys.stderr )
        print( '       -h, --help', file=sys.stderr )
        print( '       -i datafile  --input=datafile', file=sys.stderr )
        print( '       -o outdir, --outdir=outdir',file=sys.stderr )
        
def get_params( argv ):
    datafile = None
    outdir = None
        
    try:                                
        opts, args = getopt.getopt( argv, 'hi:o:',
                                    ['help','input=','outdir='] )
            
    except getopt.GetoptError:           
        self.usage()                          
        sys.exit(2)  
                   
    for opt, arg in opts:                
        if opt in ( '-h', '--help' ):      
            self.usage()                     
            sys.exit(0)
        elif opt in ( '-i', '--input' ):
            datafile = arg
        elif opt in ( '-o', '--outdir' ):
            outdir = arg
        else:
            self.usage()                     
            sys.exit(1)

    if datafile == None:
        print( 'cluster_labels: datafile is missing...exiting' )
        sys.exit(1)
    if outdir == None:
        print( 'cluster_labels: outdir is missing...exiting' )
        sys.exit(1)

    return datafile, outdir

####################################################################
# command line user entry point 
####################################################################
if __name__ == '__main__':  

    dataf,outd = get_params( sys.argv[1:] )
    print( dataf, outd )
    cluster( dataf, outd )
