#! /usr/bin/env /usr/bin/python3

#  run_diff.py
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

# compare different cluster renders

import os
import sys
import math
import numpy as np

from render_pack import render
from show_diff import compare_files
from npy_source_pack import npy_source
from cluster_weather_labels import cluster

diff_out = './2019-GA/DIFF1/'
lut_dir = './LUTs_2017-2021_5x5_4_00724_3_first/'
tally_name = diff_out + 'tally_diffs.txt'
pcent_thresh = 4           # set to math.inf for no retries
nfails = 2                 # number of threshold failures before continuing

# training instances to compare
dirs = [ './2019-GA/SOM_5x5_4_00724_3_02/',
         './2019-GA/SOM_5x5_4_00724_3_03/',
         './2019-GA/SOM_5x5_4_00724_3_04/',
         './2019-GA/SOM_5x5_4_00724_3_05/']

# ----------------------------------------------------------------------------

# make a standard 1-to-1 label LUT
nclass = 25
stdlut = np.empty( (nclass),dtype=np.uint8 )
for i in range( nclass ):
    stdlut[i] = i
    
# retrieve a LUT from file
def get_lut( i, j ):
    
    if i == j :
        print( 'using the standard look-up table', file=sys.stderr, flush=True )
        lut = stdlut
    else:
        # grab run number for both iterations
        tag1 = dirs[i][-3:-1]
        tag2 = dirs[j][-3:-1]
        lname = lut_dir + tag1 + '-' + tag2 + '.lut'
        
        # get look-up table from file:
        print( 'getting the look-up table from file:', lname,                               
               file=sys.stderr, flush=True )
        
        lfile = open( lname, 'r' )
        line = lfile.readline()
        lfile.close()
        
        items = line.split( ',' )
        lut = np.empty( (len( items )),dtype=np.uint8 )
        
        # load the look up table
        for l in range( len(items) ):
            lut[l] = int( items[l] )
            
    return lut

def check_dirs( dirs ):
    
    # check if input directories exist
    for i in range( len( dirs ) ):
        if not os.path.isdir( dirs[i]):
            print( 'run_diff: dir:', dirs[i], ' does not exist...exiting',
                   file=sys.stderr )
            sys.exit( 1 )

def transcribe_weather( i, j, outdir ):
    
    # transcribe first year's xxxlabels.npy (weather classes) for comparison
    # use LUTs from cramer run on training SOMS
    # run lut on first SOM data to transcribe classes to second SOM
    # LUTs directory does not have self LUT
    # TODO: make self lut in LUTs

    lut = get_lut( i, j )
            
    # get first years data labels file and transcribe classes using lut
    fname = dirs[i] + '2019_labels.npy'  # TODO:change this for different trainings
    src1.params.filepath = fname
    src1.run()
    
    # input is float;  make byte for transcription
    # then make back to float so clustering is done on same type
    # (does it matter?, seems to)
    print( 'transcribing',fname,'... ',file=sys.stderr, flush=True, end='')
    transcribed = lut[ src1.sink.astype( np.uint8 )].astype(np.float32)
    print( 'done.', file=sys.stderr, flush=True )

    # save to file
    tname = outdir + 'transcribed_labels.npy'
    transcribed.dump( tname )

def copy_files( i, j, outdir ):
    
    # copy cluster renders to out directory
    command =  'cp ' + dirs[i] + 'cluster.jpg ' + outdir + 'first_cluster.jpg' 
    os.system( command )
        
    command =  'cp ' + dirs[j] + 'cluster.jpg ' + outdir + 'second_cluster.jpg' 
    os.system( command )
        
    command =  'cp ' + dirs[j] + 'cluster.npy ' + outdir + 'second_cluster.npy' 
    os.system( command )

def transcribe_climate( i, j, outdir ):
    
    # put the cluster render into a buffer
    src1.params.filepath = outdir + 'cluster.npy'
    src1.run()

    # get the clusters' SOMs
    filepath1 = outdir + 'cluster.labels'
    filepath2 = dirs[j] + 'cluster.labels'

    # read cluster labels files and make similarity (lut) map
    clut = compare_files( filepath1, filepath2 )

    # read the second cluster image
    src2.params.filepath = outdir + 'second_cluster.npy'
    src2.run()
    
    # transcribe second cluster image using lut from compare_files
    second = clut[ src2.sink ].astype( np.uint8 )
    second.dump( outdir + 'transcribed_cluster.npy' )                                       

    # make a jpeg image of cluster
    rndr.source = second
    rndr.readlut( './luts/sixteenthbow.lut' )
    rndr.params.filepath = outdir + 'transcribed_cluster.jpg'
    rndr.run()

    # take the difference between clusters
    diff = src1.sink - second
    mask = (diff != 0)             # mask is logical False, True

    # highlight pixels in white that are different
    out = np.zeros( src1.sink.shape, dtype=np.uint8 )
    np.putmask( out, mask, 255 )
    count = np.count_nonzero( out )

    rndr.source = out
    rndr.readlut( './luts/rainbow.lut' )
    rndr.params.filepath = outdir + 'diff_pixels.jpg'  
    rndr.run()

    # number pixels = nx*ny
    npix = src1.sink.shape[0]*src1.sink.shape[1]
    pcent = (count/float(npix))*100

    # report to terminal
    ostr = 'num of diff pixels=' + '%d,'%count + ' % diff=' + '%.2f'%pcent
    print( ostr, file=sys.stderr, flush=True )        

    return count, pcent

def print_fail( fail_count, pcent ):
    
    # report to terminal
    print( '###################################################',
           file=sys.stderr, flush=True )
    
    print( '# run_diff: per cent error is too high:', '%.2f'%pcent,
           file=sys.stderr, flush=True )
    print( '# re-running comparison.', file=sys.stderr, flush=True )
    print( '# fail count=', fail_count, file=sys.stderr, flush=True )
    
    print( '###################################################',
           file=sys.stderr, flush=True )

    if fail_count >= nfails:
        print( '# giving up this comparison, continuing...',
               file=sys.stderr, flush=True )
        print( '###################################################',
               file=sys.stderr, flush=True )

def setup():
    
    # check if input directories exist
    check_dirs( dirs )
    
    # create output directory if needed
    if not os.path.isdir( diff_out ):
        os.mkdir( diff_out );          

# -----------------------------------------------------------------------------

# main

# check input directories and make output directory
setup()

# instantiate operators
rndr = render.render( 'render' )
src1 = npy_source.npy_source( 'npy_source' )
src2 = npy_source.npy_source( 'npy_source' )

# make difference tally out file
tfile = open( tally_name, 'w' )
tfile.write( 'dirs,num_diff_pixels,% diff\n' )
tfile.flush()

fail_count = 0                    # keep track of threshold failures

# compare ETo climate instances and calculate pixel differences
for i in range( len( dirs) ):

    j = i + 1     # target data index

    # use while loop n order to reset j counter in case of
    # unusual pixel difference error.                                         
    # python does not allow counter reset inside a loop   
    while j < len( dirs ):

        # grab run number for both iterations
        tag1 = dirs[i][-3:-1]
        tag2 = dirs[j][-3:-1]

        # create output directory
        outdir = diff_out + tag1 + '-' + tag2 + '/'
        if not os.path.isdir( outdir ):  
            os.mkdir( outdir );          

        # copy files to output directory
        copy_files( i, j, outdir )

        # transcribe first years ETo weather class values (xxxlabels.npy)
        transcribe_weather( i, j, outdir )
        
        # cluster the transcribed data; renders ETo 'climate'
        # produces outdir/cluster.npy outdir/cluster.labels outdir/cluster.jpg
        cluster( outdir + 'transcribed_labels.npy', outdir )

        # now transcribe cluster SOM and take difference for ETo climate comparison
        count, pcent = transcribe_climate( i, j, outdir )

        # re-run comparison if greater than a given percent difference threshold
        if pcent > pcent_thresh:
            fail_count += 1
            
            print_fail( fail_count, pcent )

            if fail_count < nfails:
                continue        # go to while, do not drop down
            
        # else report
        tfile.write( tag1 + '-' + tag2 + ',%d,'%count + '%.2f\n'%pcent )
        tfile.flush()
        
        j += 1
        fail_count = 0

    # end while j  
# end for i

tfile.close()
print( 'run_diff.py done', file=sys.stderr, flush=True )
