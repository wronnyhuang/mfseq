# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:23:06 2019

@author: Miguel Perez
"""
import os
from os import makedirs
from os.path import join, basename, dirname, exists
from copy import deepcopy as deepcopy
import pickle

import numpy as np
from numpy import linalg as LA
from sklearn import preprocessing

import joblib
from joblib import dump, load
import time

dataroot = 'C:/DBMM/graph_data'
output = 'C:/DBMM/scaled_graph_data'

makedirs(output, exists_ok=True)


##########
# Month based activity tag
##########
# for Months m0 m1 m2 m3 m4
# 1. find out which nodes are off for m0 m1 m2 m3, leave m4 for test set
# 2. remove the indexed nodes that are off for all 4 so that we can solve for W for those combined objects, leaving m4 for test
# 3. Add column at the end that gives a 1 iff all prior features are zero for that month, else 0
# 4. then do unit normalization using L2 unit normalization for each node
# 5. Hoyer Research paper states that fixed L2, while adjusting  alpha L1 penalty constant can guarantee
#   "better sparseness". We'll want to check that claim for our data

################################
def sum_stmt_(row_vec):
  if np.sum(row_vec[1:]) != 0:
    answer = 1  # active for 1 month
  else:
    answer = 0  # not active for 1 month
  return answer


def sum_stmt(row_vec):
  if np.sum(row_vec) > 0:
    answer = 0  # active at some point over 4 months
  else:
    answer = 1  # not active for 4 months
  return answer


################################

X_filename = 'M' + str(0) + '_for_nmf.joblib'
with open(join(dataroot, X_filename), 'rb') as f:
  X = joblib.load(f)
index_array_0 = X[:, 0]
data_matrix_activity_tag_0 = np.apply_along_axis(sum_stmt_, 1, X)  # vector activty tag

############
X_filename = 'M' + str(1) + '_for_nmf.joblib'
with open(join(dataroot, X_filename), 'rb') as f:
  X = joblib.load(f)
data_matrix_activity_tag_1 = np.apply_along_axis(sum_stmt_, 1, X)

############
X_filename = 'M' + str(2) + '_for_nmf.joblib'
with open(join(dataroot, X_filename), 'rb') as f:
  X = joblib.load(f)
data_matrix_activity_tag_2 = np.apply_along_axis(sum_stmt_, 1, X)

############
X_filename = 'M' + str(3) + '_for_nmf.joblib'
with open(join(dataroot, X_filename), 'rb') as f:
  X = joblib.load(f)
data_matrix_activity_tag_3 = np.apply_along_axis(sum_stmt_, 1, X)

stacked_activity_tag = np.stack(
  (data_matrix_activity_tag_0, data_matrix_activity_tag_1, data_matrix_activity_tag_2, data_matrix_activity_tag_3),
  axis=1)
stacked_activity_tag = np.apply_along_axis(sum_stmt, 1, stacked_activity_tag)  # vector of activity tag over 4 months
# stacked_activity_tag[100002] #is 0 for 1 0 1 1
# stacked_activity_tag[100001] #is 0 for 1 1 1 1

# get a list of node indices that should be removed from the datacube
vision_function = lambda input_array, index: index if input_array[index] == 1 else -1
prune_by_row_index = [vision_function(stacked_activity_tag, index) for index in range(stacked_activity_tag.shape[0])]
prune_by_row_index = [x for x in prune_by_row_index if x != -1]  #

####################
# Now we add stacked activity_tag as a new columns to the data_matrix X_0
# and then delete from it the rows belonging to prune_by_row_index
# then unit normalize by row with L2 norm
# and then save X_new to disk
###################


# Now do this once for X_0, X_1, X_2, X_3

# Month_index = 0
for Month_index in range(4):
  X_filename = 'M' + str(Month_index) + '_for_nmf.joblib'
  with open(join(dataroot, X_filename), 'rb') as f:
    X = joblib.load(f)
  X_new = np.hstack((X, np.reshape(stacked_activity_tag, (stacked_activity_tag.shape[0], 1))))
  X_new = np.delete(X_new, prune_by_row_index, 0)
  DBMM_nodes = X_new[:, 0]

  # We need to drop first column before unit normalizing, so we'll want to save it as a node index list
  dump(DBMM_nodes, join(output, 'DBMM_M' + str(Month_index) + '_nodes_by_index' + '.joblib'))
  # pickle.dump( model_results_by_Fro_norm, open(outputroot+'M'+str(0) + '_FULL_role_by_loss_list', 'wb' ) )
  X_new = preprocessing.normalize(X_new[:, 1:], norm='l2')

  # find the index of the largest contribution to the unit norm
  np.where(X_new[0] == np.amax(X_new[0]))
  dump(X_new, join(output, 'DBMM_M' + str(Month_index) + '_data_matrix_unit_norm' + '.joblib'))

# Done!