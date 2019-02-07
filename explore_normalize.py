from os.path import join, basename, dirname, exists
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout
import numpy as np
import numpy as np
import tensorflow as tf
import pickle
import os
from os import makedirs
import joblib
from shutil import rmtree
from time import sleep

home = os.environ['HOME']
dataroot = join(home, 'datasets/everbank_graph')
imageroot = join(dataroot, 'tmp')
makedirs(imageroot, exist_ok=True)

datacube = []
for i in range(3,3+5):
  with open(join(dataroot, 'M'+str(i)+'_for_nmf.joblib'), 'rb') as f:
    raw = joblib.load(f)
  datacube.append(raw)

# Time x Node x Feature
datacube = np.stack(datacube, axis=0)
ntime, nnode, nfeat = datacube.shape

## max, mean, and nonzero of each feature marginal
idxfeat = np.arange(0, nfeat)
# featstat = datacube.mean(axis=(0,1))
featstat = np.count_nonzero(datacube, axis=(0,1))
# featstat = featstat / (ntime * nnode)
featstat = np.log10(featstat)
plot(idxfeat, featstat)
xlabel('feature id')
ylabel('log value')
title('count nonzero')
savefig('exploratory/nonzero.pdf')
show()
close('all')

## plot feature marginal histogram
idxlog = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

for idxfeat in range(nfeat):

  featMarginal = datacube[:,:,idxfeat].reshape(-1)
  featMarginal = featMarginal[featMarginal!=0]

  # skip features having no nonzero values
  if not any(featMarginal):
    print(idxfeat, 'all zeros, skipping')
    continue

  # possibly swithc to log scale and then normalize
  normalized = np.log10(featMarginal) if idxfeat in idxlog else featMarginal
  normalized = ( normalized - normalized.mean() ) / normalized.std()

  # plot
  figure(figsize=(7.5,3))
  suptitle('feature id: '+str(idxfeat))

  subplot(1,3,1)
  hist(featMarginal, 50, color='red')
  xlabel('feature value')
  title('linear scale')

  subplot(1,3,2)
  hist(np.log10(featMarginal), 50)
  xlabel('log feature value')
  title('log scale')

  subplot(1,3,3)
  logstr = 'log ' if idxfeat in idxlog else ''
  hist(normalized, 50, color='green')
  xlabel('new feature value')
  title(logstr+'normalized')

  tight_layout(pad=1.5)
  savefig(join(imageroot, str(idxfeat)+'.jpg'))
  close('all')
  sleep(.3)
  print(idxfeat)


##
os.system('python make_gif.py '+imageroot)
rmtree(imageroot)



