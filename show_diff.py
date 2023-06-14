#! /usr/bin/env /usr/bin/python3

#  show_diff.py
# 
#  Copyright (C) 2020-2021 Scott L. Williams.
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

# show the pixel difference between images

import sys
import math
import numpy as np
import itertools

dirpath1 = './2019/SOM_5x5_6_01_3_1/'
dirpath2 = './2020/SOM_5x5_6_01_3_1/'

# ----------------------------------------------------------------------------

# exhaustively run through LUT permutations for least pixel difference between clusters
# impresivelly effective. however the brute force approach takes days
# ie. for a lut of 12 elements it takes 12! (479,001,600) iterations
def compare_luts( imagepath1, imagepath2, nlabels ):
    
    print( 'comparing images: ', imagepath1, 'and', imagepath2,
           file=sys.stderr, flush=True )

    # first image
    src1 = npy_source.npy_source( 'npy_source' )
    src1.params.filepath = imagepath1
    src1.run()

    # second image
    src2 = npy_source.npy_source( 'npy_source' )
    src2.params.filepath = imagepath2
    src2.run()
    
    if src1.sink.shape != src2.sink.shape:
        print( 'compaee_luts: image sizes do not match..exiting.',
               file=sys.stderr, flush=True )
        sys.exit( 2 )

    # generate unique LUT values, ie. 0,1,2,3,...,N-1
    invalues = [i for i in range(nlabels) ] # find unique permutations for this array

    # test each LUT for minimum pixel difference
    mindiff = math.inf
    minlut = None
    lut = None
    i = 0
    for l in itertools.permutations( invalues ):

        if i%100000 == 0:
            print( 'i=', i, 'mindiff=', mindiff, 'lut:', minlut, 'last lut:', lut,
                   file=sys.stderr, flush=True )
            
        # get the LUT permutation
        lut = np.array( l )
        
        # transcribe labels from first run for comparison
        second = lut[ src2.sink ].astype( np.uint8 )

        # take the difference
        diff = src1.sink - second
        mask = (diff != 0)             # mask is logical False, True

        # highlight pixels in white that are different
        out = np.zeros( src1.sink.shape, dtype=np.uint8 )
        np.putmask( out, mask, 255 )

        count = np.count_nonzero( out )
        if count < mindiff:
            minlut = lut;
            mindiff = count
            
        i +=1
        
    return minlut

# use some metric to measure closeness
def distance( neurons1, neurons2 ):

    nlabels = len( neurons1 )
    if nlabels != len( neurons2 ):
        print( 'number of labels do not match ... exiting.',
               file=sys.stderr, flush=True )

    ndims = len( neurons1[0] )
    if ndims != len( neurons2[0] ):
        print( 'number of neuron dimensions do not match ... exiting.',
               file=sys.stderr, flush=True )

    # allocate array of all distance
    arr = [ [0 for i in range( nlabels ) ] for j in range( nlabels ) ]

    # array of vector distances
    for j in range( nlabels ):
        for i in range( nlabels ):
            # euclidean
            arr[j][i] = math.sqrt(sum(pow(a-b,2) for a, b in zip(neurons1[j],neurons2[i]) ) )

            # manhattan
            #arr[j][i] = sum(abs(a-b) for a,b in zip(neurons1[j], neurons2[i]) )
            
    '''
    for j in range( nlabels ):
        for i  in range( nlabels ):
            print( '%5.1f '%arr[j][i], end='' )
        print( end='\n' )
    '''
    
    labelmap = []

    for l in range( nlabels ):
    
        # find next minimum and unique indices
        mindist = math.inf
        for j in range( nlabels ):
            for i in range( nlabels ):
                found = False
                for k in range( l ):
                    if i == labelmap[k][0]:
                        found = True
                    if j == labelmap[k][1]:
                        found = True
                if not found and arr[j][i] < mindist:
                    mindist = arr[j][i]
                    ij = j
                    ii = i

        labelmap.append( [ ii, ij, mindist ] )

    labelmap = sorted( labelmap )
    
    lut = np.empty( (nlabels), dtype=np.int8 )
    for i  in range(nlabels ):
        lut[i] = labelmap[i][1]

    return lut

def get_label_header( labels ):

    # read header
    found = False
    for line in labels:
        if line.find( 'NEURONS' ) != -1:
            found = True
            break
        
    if not found:
        return None, None

    # read next line for number of labels (neurons) and dimensions
    nlabels,ndims = labels.readline().split()
    nlabels = int( nlabels )
    ndims = int( ndims )

    return nlabels, ndims


def compare_files( filepath1, filepath2 ):
    
    print( 'comparing files: ', filepath1, 'and', filepath2,
           file=sys.stderr, flush=True )

    # first labels file
    labels1 = open( filepath1, 'r' )
    nlabels1, ndims1 = get_label_header( labels1 )
    if nlabels1 == None:
        print( 'could not find NEURONS flag in ', filepath1, ' exiting.',
               file=sys.stderr, flush=True )
        sys.exit( 2 )

    # second file
    labels2 = open( filepath2, 'r' )
    nlabels2, ndims2 = get_label_header( labels2 )
    if nlabels2 == None:
        print( 'could not find NEURONS flag in ', filepath2, ' exiting.',
               file=sys.stderr, flush=True )
        sys.exit( 2 )

    if nlabels1 != nlabels2:
        print( 'number of labels do not match...exiting',
               file=sys.stderr, flush=True )
        sys.exit( 2 )
    
    if ndims1 != ndims2:
        print( 'number of dimensions do not match...exiting',
               file=sys.stderr, flush=True )
        sys.exit( 2 )
        
    print( 'reading vectors... ', file=sys.stderr, flush=True )

    # actually 2D arrays
    neurons1 = []
    neurons2 = []

    # populate arrays
    for i in range( nlabels1 ):

        # read and load the weights of first file
        line1 = labels1.readline().split()
 
        # get the neuron weights (vector feature space)
        neurons1.append([])
        for j in range( 1, ndims1+1 ):       # skip grey level label
            neurons1[i].append( float( line1[j] ) )
     
        # second file
        line2 = labels2.readline().split()
        neurons2.append([])
        for j in range( 1, ndims2+1 ):       # skip grey level label
            neurons2[i].append( float( line2[j] ) )
   
    # compare arrays

    # get vector similarities
    return distance( neurons1, neurons2 )

#------------------------------------------------
if __name__ == '__main__':
    
    from npy_source_pack import npy_source
    from render_pack import render

    rndr = render.render( 'render' )

    # main
    filepath1 = dirpath1 + 'cluster.labels'
    filepath2 = dirpath2 + 'cluster.labels'

    # read cluster labels files and make similarity (lut) map
    lut = compare_files( filepath1, filepath2 )
    print( lut, file=sys.stderr, flush=True )

    # use this for exhaustive approach
    #lut = compare_luts( dirpath1 + 'cluster.npy', dirpath2 + 'cluster.npy', 12 )
    #print( lut, file=sys.stderr, flush=True )
    #sys.exit( 0 )

    # show the difference
    
    # first source
    src1 = npy_source.npy_source( 'npy_source' )
    src1.params.filepath = dirpath1 + 'cluster.npy'

    # second source
    src2 = npy_source.npy_source( 'npy_source' )
    src2.params.filepath = dirpath2 + 'cluster.npy'

    # read the first cluster image
    print( 'reading first datafile: ' + src1.params.filepath + '...',
           file=sys.stderr, flush=True, end='' )
    src1.run()
    print( 'done', file=sys.stderr, flush=True )

    # read the second cluster image
    print( 'reading second datafile: ' + src2.params.filepath + '...',
           file=sys.stderr, flush=True, end='' )
    src2.run()
    print( 'done', file=sys.stderr, flush=True )

    # transcribe labels from first run for comparison
    second = lut[ src2.sink ].astype( np.uint8 )

    rndr.source = second
    rndr.readlut( '/home/agrineer/eto_study/scripts/luts/sixteenthbow.lut' )
    rndr.params.filepath = 'transcribed.jpg'  
    rndr.run()

    # take the difference
    diff = src1.sink - second
    mask = (diff != 0)             # mask is logical False, True

    # highlight pixels in white that are different
    out = np.zeros( src1.sink.shape, dtype=np.uint8 )
    np.putmask( out, mask, 255 )

    count = np.count_nonzero( out )
        
    # number pixels =  nx*ny
    npix = src1.sink.shape[0]*src1.sink.shape[1]
    pcent = (count/float(npix))*100

    # report to terminal
    ostr = 'num of diff pixels=' + '%d,'%count + ' % diff=' + '%.2f'%pcent
    print( ostr, file=sys.stderr, flush=True )

    rndr.source = out
    rndr.readlut( '/home/agrineer/eto_study/scripts/luts/rainbow.lut' )
    rndr.params.filepath = 'diff_pixels.jpg'  
    rndr.run()
