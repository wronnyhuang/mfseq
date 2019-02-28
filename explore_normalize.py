import gzip
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
from sklearn.preprocessing import normalize

home = os.environ['HOME']
dataroot = join(home, 'datasets/graph_data_explore')
imageroot = join(dataroot, 'images')
makedirs(imageroot, exist_ok=True)

datacube = []
for i in range(3,3+5):
  with open(join(dataroot, 'M'+str(i)+'_for_nmf.joblib'), 'rb') as f:
    raw = joblib.load(f)
  datacube.append(raw)

# Time x Node x Feature
datacube = np.stack(datacube, axis=0)

# save the shape of the datacube
ntime, nnode, nfeat = datacube.shape

## max, mean, and nonzero of each feature marginal
idxfeat = np.arange(0, nfeat)
# featstat = datacube.mean(axis=(0,1))
featstat = np.count_nonzero(datacube, axis=(0,1))
# featstat = featstat / (ntime * nnode)
featstat = np.log10(featstat)
# plot(idxfeat, featstat)
# xlabel('feature id')
# ylabel('log value')
# title('count nonzero')
# savefig('exploratory/nonzero.pdf')
# show()
# close('all')

## plot feature marginal histogram
idxlog = [1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] # indices of features we should log scale
norm_mean = np.zeros(nfeat, dtype=float)
norm_std = np.zeros(nfeat, dtype=float)
for idxfeat in range(nfeat):

  # flatten data
  featMarginal = datacube[:,:,idxfeat].reshape(-1)

  # skip features having no nonzero values
  featMarginal = featMarginal[featMarginal != 0]
  if not any(featMarginal):
    print(idxfeat, 'all zeros, skipping')
    norm_mean[idxfeat] = np.nan
    norm_std[idxfeat] = np.nan
    continue

  # possibly swithc to log scale and then normalize
  def logpositive(input):
    output = np.log10(input)
    output = output - np.min(output) + 1e-6 # shift values such that the minimum value is 0, not inf
    return output

  # scaling function is either log or identity
  scalingfun = logpositive if idxfeat in idxlog else lambda x: x
  scaled = scalingfun(featMarginal)

  # save mean value in case you want to denormalize it in the future
  norm_mean[idxfeat] = scaled.mean()

  # get all nonzero indices
  nonzeroIdx = np.nonzero(datacube[:, :, idxfeat])

  # update each nonzero datacube entry to be scaled-normalized
  datacube[nonzeroIdx[0], nonzeroIdx[1], idxfeat] = scalingfun(datacube[nonzeroIdx[0], nonzeroIdx[1], idxfeat]) / norm_mean[idxfeat]

  ## plot
  # # the normalized marginal
  # normalized = scaled / norm_mean[idxfeat]
  #
  # figure(figsize=(7.5,3))
  # suptitle('feature id: '+str(idxfeat))
  #
  # subplot(1,3,1)
  # hist(featMarginal, 50, color='red')
  # xlabel('feature value')
  # title('linear scale')
  #
  # subplot(1,3,2)
  # hist(np.log10(featMarginal), 50)
  # xlabel('log feature value')
  # title('log scale')
  #
  # subplot(1,3,3)
  # logstr = 'log ' if idxfeat in idxlog else ''
  # hist(normalized, 50, color='green')
  # xlabel('new feature value')
  # title(logstr+'normalized')
  #
  # tight_layout(pad=1.5)
  # savefig(join(imageroot, str(idxfeat)+'.jpg'))
  # close('all')
  # sleep(.3)
  # print(idxfeat)

##

# remove the first feature since that is just the index of the node
datacube = datacube[:, :, 1:]

# remove totally inactive nodes
active = [np.any(datacube[:,i,:]) for i in range(nnode)]
datacube = datacube[:,active,:]

# append onto the cube the "node-inactive-this-month feature", and also unit normalize
normalize = lambda x: x / np.linalg.norm(x)
datacube = np.array([[normalize(np.append(node, (0 if np.any(node) else 1))) for node in month] for month in datacube])

# save the datacube
with gzip.open('graph_data_cube.pkl', 'wb') as f:
  pickle.dump(datacube, f)

## table-ize the sparse cube into highly compressed form

# get nonzero indices and their corresponding values in datacube
indices = np.argwhere(np.logical_and(datacube > -1e8, np.abs(datacube) > 1e-8))
indices = indices.transpose()
labels = datacube[indices[0], indices[1], indices[2]]

with gzip.open('graph_data_table.pkl', 'wb') as f:
  pickle.dump((indices, labels, datacube.shape), f)

with open('norm_mean_std.pkl', 'wb') as f:
  pickle.dump((norm_mean, norm_std, idxlog), f)

# os.system('python make_gif.py '+imageroot)
# rmtree(imageroot)



