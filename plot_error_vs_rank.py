import numpy as np
from comet_ml import API
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim

api = API(rest_api_key='W2gBYYtc8ZbGyyNct5qYGR2Gl')
experiments = api.get('wronnyhuang/nmf-sweeprank')

ranks = []
errors = []
for experiment in experiments:

  # get the rank
  rank = int(api.get_experiment_parameters(experiment, 'rank')[0])
  ranks.append(rank)

  # get the minimum test error
  metrics = {m.pop('name'): m for m in api.get_experiment_metrics(experiment)}
  error = float(metrics['test/crit']['valueMin'])
  error = np.sqrt( error * 767176 ) # multiply by test set size to undo averaging and take sqrt to complete the norm
  errors.append(error)

plot(ranks, errors, '.', markersize=8)
title('rank vs error')
xlabel('rank')
ylabel('error (frobenius norm)')
show()
