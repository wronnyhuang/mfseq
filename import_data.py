from os.path import join, basename, dirname, exists
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig
import numpy as np
import numpy as np
import tensorflow as tf
import pickle
import os
import joblib

home = os.environ['HOME']
dataroot = join(home, 'datasets/everbank_graph')

datacube = []
for i in range(3,3+5):
  with open(join(dataroot, 'M'+str(i)+'_for_nmf.joblib'), 'rb') as f:
    raw = joblib.load(f)
  datacube.append(raw)

datacube = np.stack(datacube, axis=0)

for k in range(datacube.shape[2]):
  featMarginal = datacube[:,:,k].reshape(-1)
  print(k, featMarginal.max(), featMarginal.mean())

featMarginal = datacube[:,:,12].reshape(-1)
featMarginal = featMarginal[featMarginal!=0]
fmlog = np.log10(featMarginal)
hist(fmlog, 20)
savefig('plot.jpg')
show()




