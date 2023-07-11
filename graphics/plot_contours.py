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

import sys
import numpy as np
import matplotlib.pyplot as plt

from npy_source_pack import npy_source
#from scale_pack import scale

#dirpath = './2021/SOM_5x5_9_02_3_1/'
dirpath = './2021/SOM_5x5_9_02_3_DIFF1/1-7/'
nlabels = 12

# numpy source
src = npy_source.npy_source( 'npy_source' )        # instantiate
src.params.filepath = dirpath + 'transcribed_cluster.npy'

# read the label image
print( 'reading datafile: ' + src.params.filepath + '...',
       file=sys.stderr, flush=True, end='' )
src.run()
print( 'done', file=sys.stderr, flush=True )

'''
# scale it for contouring
sc = scale.scale( 'scale' )                        # instantiate
sc.params.sx = 6.0
sc.params.sy = 6.0
sc.source = src.sink
sc.run()

# contour the scaled image
image = sc.sink[:,:,0]
'''
image = src.sink[:,:,0]

# set up plot
fig,ax = plt.subplots(1,1)
levels = np.arange(1, nlabels, 1.0)
CS = ax.contour( image, levels, linewidths=.05, colors='black' )
#ax.clabel( CS, inline=True, fontsize=10 )

plt.ylim( 171, 0 ) # apparently vertically flips the image

plt.title( 'Source file: ' + src.params.filepath )
plt.show()
