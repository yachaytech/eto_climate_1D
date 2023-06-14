#! /usr/bin/env /usr/bin/python3

#  plot_pixels_hist.py
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dirpath = './2020/SOM_5x5_6_01_3_2/'
srcfile = dirpath + 'hist_points.csv'

# first read the sample file
infile = open( srcfile )

srcfile = infile.readline()              # get source filename

values = []                              # array of variable value arrays
xpixpos = []                             # array of sampled pixel x positions
for line in infile:
    
    items = line.split( ',' )            # parse out values from text line
    xpixpos.append( int( items[0] ) )    # grab pixel x position
    
    varvalues = []                       # array of variable values 
    for i in range(2,len(items)):      
        varvalues.append( float(items[i].strip()) )
    values.append( varvalues )
    
    #values.append( varvalues[::-1] )     # reverse order of values to make
                                          # graph more intiutive when rotated

# set up plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# lengths colors has to match number of samples
# TODO: find way to create colors
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple','tab:brown' ]

for i in range( len(values) ):

    # plot the bar graph 
    xs = range( len(values[i]) )
    ax.bar( xs, values[i],  zs=xpixpos[i], zdir='y', color=colors[i], alpha=0.7)

ax.set_xlabel('\nClass Migration Histogram',linespacing=1)
ax.set_ylabel('\nPixel X position\nY=0',linespacing=1)
ax.set_zlabel('\nNumber of Days',linespacing=1)

plt.title( 'Source file: ' + srcfile )
plt.show()
