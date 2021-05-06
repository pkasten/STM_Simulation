
import pySPM
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.morphology import binary_erosion, disk
#%matplotlib inline

import os
from IPython import display

#from pySPM_data import get_data



filename = "HOPG-gxy1z1-p2020.sxm"
S = pySPM.SXM(filename)
S.list_channels()
fig, ax = plt.subplots(1,1,figsize=(14,7))
z = S.get_channel('Z')
z.show(ax=ax)
#plt.show()
plt.savefig("file1.png")
#p = S.get_channel('Current').show(ax=ax[1], cmap='viridis')



topo2 = z.correct_lines(inline=False)
topo3 = z.correct_plane(inline=False)

fig, ax = plt.subplots(1,3,figsize=(21, 7))
z.show(ax=ax[0])
ax[0].set_title("Original image")
topo2.show(ax=ax[1])
ax[1].set_title("Corrected by lines")
topo3.show(ax=ax[2])
ax[2].set_title("Corrected by slope");
#plt.show()
plt.savefig("file2.png")

fig = plt.figure(figsize=(7,7))
import copy
topo4 = copy.deepcopy(z) # make deepcopy of object otherwise you will just change the original
topo4.correct_median_diff()
topo4.show()
plt.savefig("file3.png")

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

topoC = topo4.filter_scars_removal(.7,inline=False)
topo4.show(ax=ax[0])
topoC.show(ax=ax[1])
for a in ax:
    for p in [(71,50),(10,13),(32,13),(80,5)]:
        a.annotate("", p, (p[0]+5, p[1]+5), arrowprops=dict(arrowstyle="->", color='w'));

plt.savefig("file4.png")

#x1 = int(input("Flat Line x1, range {}-{}: ".format(0, np.shape(topoC.pixels)[0] - 1)))
#y1 = int(input("Flat Line y1, range {}-{}: ".format(0, np.shape(topoC.pixels)[1] - 1)))
#x2 = int(input("Flat Line x2, range {}-{}: ".format(0, np.shape(topoC.pixels)[0] - 1)))
#y2 = int(input("Flat Line y2, range {}-{}: ".format(0, np.shape(topoC.pixels)[1] - 1)))
x1 = 10
x2 = 10
y1 = 10
y2 = 200


topoD = topoC.corr_fit2d(inline=False).offset([[x1, y1, x2, y2]]).filter_scars_removal()

mask0 = topoD.get_bin_threshold(.1, high=False)
mask1 = binary_erosion(mask0, disk(3))
topoD2 = topoC.corr_fit2d(mask=mask1, inline=False).offset([[x1, y1, x2, y2]]).filter_scars_removal().zero_min()

fig, ax = plt.subplots(2, 3, figsize=(21, 14))
ax = np.ravel(ax)
topoC.show(ax=ax[0])
ax[0].set_title("Image: median_diff correction")
topoD.show(ax=ax[1], sig=1.5);
ax[1].set_title("Polynomial correction on whole image")
ax[2].imshow(mask0);
ax[2].set_title("Mask: binary thersholding")
ax[3].imshow(mask1);
ax[3].set_title("Mask: background selection");
topoD2.show(ax=ax[4], pixels=True);
ax[4].set_title("Polynomial correction on background mask");
topoD2.show(ax=ax[5], adaptive=True)

plt.savefig("file5.png")


fig, (ax, ax2) = plt.subplots(2, 3, figsize=(21, 14))
topoD.show(ax=ax[0], cmap='gray', title="color map=\"gray\"")
topoD.show(ax=ax[1], sig=2, title="standard deviation=2")
topoD.show(ax=ax[2], adaptive=True, title="Adaptive colormap")
mini = np.min(topoD.pixels)
maxi = np.max(topoD.pixels)
print(mini)
print(maxi)
topoD.show(ax=ax2[0], dmin=mini, dmax=maxi, cmap='gray', title="raise the lowest value for the colormap of +40nm")
topoD.show(ax=ax2[1], dmin=-3e-9, dmax=3e-9, cmap='gray',title="raise lower of +30nm and highest of -30nm")
topoD.show(ax=ax2[2], pixels=True, title="Set axis value in pixels");

plt.savefig("file6.png")



fig, ax = plt.subplots(1,2,figsize=(10,5))

topoD2.plot_profile(280,250,400,208,ax=ax[1],img=ax[0])
topoD2.show(ax=ax[0],pixels=True)
pySPM.utils.Ydist(plt.gca(), .14, 1.22, 27, unit="nm")
plt.tight_layout()

plt.savefig("file7.png")

fig, ax = plt.subplots(1,3,figsize=(21,5))

x1 = int(input("Flat Line x1, range {}-{}: ".format(0, np.shape(topoC.pixels)[0] - 1)))
y1 = int(input("Flat Line y1, range {}-{}: ".format(0, np.shape(topoC.pixels)[1] - 1)))
x2 = int(input("Flat Line x2, range {}-{}: ".format(0, np.shape(topoC.pixels)[0] - 1)))
y2 = int(input("Flat Line y2, range {}-{}: ".format(0, np.shape(topoC.pixels)[1] - 1)))

topoD2.show(ax=ax[0])
topoD2.plot_profile(x1, y1, x2, y2, ax=ax[1], img=ax[0], pixels=False);
topoD2.plot_profile(x1, y1, x2, y2, ax=ax[2], pixels=False);
ax[2].set_title("ztransf can be used to adj. the z-scale.\n Here in nm with the minimum at 0");
topoD2.ztransf()
ax[2].grid();
ax[2].yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator()) # set minor ticks every units
ax[2].xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator()) # set minor ticks automatically
ax[2].grid(axis='both',which='minor',alpha=.2); # display the minor grid every nm vertically
plt.tight_layout();

plt.savefig("file8.png")