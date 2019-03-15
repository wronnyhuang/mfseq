from comet_ml import Experiment
import tensorflow as tf
import numpy as np
import argparse
import os
from os.path import join, basename, dirname, exists
import pickle
import gzip
import joblib
import torch.utils.data
from time import time
from utils import maybe_download, timenow
from functools import reduce
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim, clim
import matplotlib.pyplot as plt

# parse terminal arguments
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-rank', default=10, type=int)
parser.add_argument('-batchsize', default=20000, type=int)
parser.add_argument('-lrnrate', default=.1, type=float)
parser.add_argument('-lrdrop', default=190, type=int)
parser.add_argument('-trancoef', default=.3, type=float)
parser.add_argument('-wdeccoef', default=1e-8, type=float)
parser.add_argument('-hdeccoef', default=1e-4, type=float)
parser.add_argument('-nnegcoef', default=1e8, type=float)
parser.add_argument('-maxgradnorm', default=1e10, type=float)
parser.add_argument('-nepoc', default=400, type=int)
parser.add_argument('-logtrain', default=2, type=int)
parser.add_argument('-logtest', default=5, type=int)
parser.add_argument('-logimage', action='store_true')
parser.add_argument('-dumpdisk', action='store_true')
args = parser.parse_args()

def p_inv(matrix):
  '''Returns the Moore-Penrose pseudoinverse'''
  s, u, v = tf.svd(matrix)
  threshold = tf.reduce_max(s) * 1e-5
  s_mask = tf.boolean_mask(s, s > threshold)
  s_inv = tf.diag(tf.concat([1. / s_mask, tf.zeros([tf.size(s) - tf.size(s_mask)])], 0))
  return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))


class Model():

  def __init__(self, args, ntime, nnode, nfeat):
    '''constructor of the model. create computation graph and build the session'''
    self.args = args
    self.build_graph(ntime, nnode, nfeat)
    self.build_sess()


  def build_graph(self, ntime, nnode, nfeat):
    '''build computational graph'''

    # inputs passed via feed dict
    with tf.name_scope('inputs'):
      self.lr = tf.placeholder(tf.float32, name='lr')
      self.X_true = tf.placeholder(tf.float32, [ntime-1, nnode, nfeat], name='X_true')
      self.x_last = tf.placeholder(tf.float32, [nnode, nfeat], name='x_last')

    # forward propagation from inputs to predictions
    with tf.name_scope('forward'):
      self.H = tf.Variable(tf.random_gamma(shape=[args.rank, nfeat], alpha=1.0), name='H')
      self.T = tf.Variable(tf.random_gamma(shape=[args.rank, args.rank], alpha=1.0), name='T')
      self.W = []
      X_comp = [] # X computed from compressed representation
      X_tran = [] # X computed from transition matrix applied to previous compressed representations

      # loop through time steps
      for t in range(ntime-1):
        w = tf.Variable(tf.random_gamma(shape=[nnode, args.rank], alpha=1.0), name='W_'+str(t))
        self.W.append(w)
        X_comp.append(tf.matmul(w, self.H, name='X_comp_'+str(t)))
        if t >= 1: # X_tran is the result of a transition
          x_tran = tf.matmul( tf.matmul(self.W[-2], tf.matmul(self.T, self.T)) + tf.matmul(self.W[-1], self.T), self.H, name='X_tran_'+str(t+1))
          X_tran.append(x_tran)

    # define loss criterion here (squared error, KL divergence, etc)
    criterion = tf.losses.mean_squared_error

    # regularization terms
    with tf.name_scope('regularizers'):
      with tf.name_scope('weightdecay'):
        self.wdec = tf.reduce_sum(tf.add_n([tf.norm(w, ord=1, axis=1)**2 for w in self.W]))
        self.hdec = tf.reduce_sum(tf.norm(self.H, ord=1, axis=0)**2)
      with tf.name_scope('nonnegativity'):
        self.nneg = tf.reduce_sum(tf.add_n([tf.nn.relu(-w)**2 for w in self.W])) + tf.reduce_sum(tf.nn.relu(-self.H)**2)
        self.nneg = self.nneg / ( (ntime-1)*nnode*args.rank + args.rank*nfeat )

    # loss terms
    with tf.name_scope('losses'):
      with tf.name_scope('compression'):
        self.comp = criterion(X_comp, self.X_true)
      with tf.name_scope('transition'):
        self.tran = criterion(X_comp[2:], X_tran[:-1])

    # optimization objective
    self.cost = tf.add_n([(1 - args.trancoef) * self.comp,
                          args.trancoef * self.tran,
                          args.hdeccoef * self.hdec,
                          args.wdeccoef * self.wdec,
                          args.nnegcoef * self.nneg], name='cost')

    # training operations
    with tf.name_scope('train_ops'):
      self.step = tf.train.get_or_create_global_step()
      opt = tf.train.AdamOptimizer(self.lr)
      grads = tf.gradients(self.cost, tf.trainable_variables())
      grads, self.gradnorm = tf.clip_by_global_norm(grads, args.maxgradnorm)
      self.trainop = opt.apply_gradients(zip(grads, tf.trainable_variables()), global_step=self.step, name='trainop')

    # test operations
    with tf.name_scope('test_ops'):
      with tf.name_scope('compression_test'):
        H_inv = p_inv(self.H)
        self.w_last = tf.matmul(self.x_last, H_inv)
        x_comp = tf.matmul(self.w_last, self.H)
        self.comptest = criterion(x_comp, self.x_last)
        self.comptestfrob = tf.norm((x_comp-self.x_last)**2)
      with tf.name_scope('transition_test'):
        self.trantest = criterion(X_tran[-1], x_comp)
        self.trantestfrob = tf.norm((x_comp-X_tran[-1])**2)


  def build_sess(self):
    '''start tf session'''
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(self.args.logdir, graph=self.sess.graph)


  def test(self, x_last, step):
    '''test on entire test set'''
    metricnodes = dict(comp=self.comptest, tran=self.trantest, compfrob=self.comptestfrob, tranfrob=self.trantestfrob)
    metrics, self.w_last_save = self.sess.run([metricnodes, self.w_last], {self.x_last: x_last})
    experiment.log_metrics(metrics, prefix='test', step=step)
    print('TEST:\tstep', step, '\tcomp', metrics['comp'], '\ttran', metrics['tran'])


  def fit(self, datacube):
    '''fit the model to the data presented in the input dataloader'''

    # split the data into train and test
    X_true = datacube[:-1, :, :]
    x_last = datacube[-1, :, :]
    print('TRAIN: training on %s time steps of data, nnode=%s nfeat=%s rank=%s' % (len(X_true), nnode, nfeat, args.rank))

    # start looping through epochs
    step = 0
    metricnodes = dict(cost=self.cost, comp=self.comp, tran=self.tran, wdec=self.wdec, hdec=self.hdec, nneg=self.nneg, gradnorm=self.gradnorm)
    for epoch in range(args.nepoc):
      dropfactor = 0.1 if step > args.lrdrop else 1
      _, metrics, step, = self.sess.run([self.trainop, metricnodes, self.step],
                                        {self.lr: args.lrnrate * dropfactor,
                                         self.X_true: X_true,
                                        })
      if np.mod(step, args.logtrain)==0:
        experiment.log_metrics(metrics, step=step)
        print('TRAIN:\tepoch', epoch, '\tstep', step, '\tcomp', metrics['comp'], '\ttran', metrics['tran'], '\tcost', metrics['cost'])
      if np.mod(step, args.logtest)==0:
        self.test(x_last, step)
        if args.logimage: self.plot()
    print('done training')


  def get_params(self):
    return self.sess.run(dict(W=self.W, H=self.H, T=self.T))


  def plot(self, ending=False):
    '''plot and save distributions'''
    params = self.get_params()
    W, H, T = self.sess.run([self.W, self.H, self.T])
    for i, w in enumerate(W):
      figname = 'distribution-W_'+str(i)
      hist(w.ravel(), 200); xlim(-.1, 2); title(figname)
      experiment.log_figure(figure_name=figname, figure=plt.gcf())
      close('all')
    figname = 'distribution-H'
    hist(H.ravel(), 200); xlim(-.1, 2); title(figname)
    experiment.log_figure(figure_name=figname, figure=plt.gcf())
    close('all')
    figname = 'distribution-T'
    hist(T.ravel(), 10); title(figname)
    experiment.log_figure(figure_name=figname, figure=plt.gcf())
    close('all')

    if ending:

      # plot distribution of values excluding zeros
      hist(datacube.ravel()[np.abs(datacube.ravel())>1e-2].ravel(), 200); xlim(-.1, 2); title('distribution-values')
      experiment.log_figure(figure_name='distribution-values', figure=plt.gcf())
      close('all')

      # plot distribution of values including zeros
      hist(datacube.ravel(), 200); xlim(-.1, 2); title('distribution-values-withzeros')
      experiment.log_figure(figure_name='distribution-values', figure=plt.gcf())
      close('all')

      # plot H matrix heatmap
      figure(figsize=(24,10))
      imshow(H); axis('image'); colorbar(); clim(0, 1)
      experiment.log_figure(figure_name='H-matrix')
      close('all')

      # plot T matrix heatmap
      figure(figsize=(10,10))
      imshow(T); axis('image'); colorbar();
      experiment.log_figure(figure_name='T-matrix')
      close('all')

      # plot W matrix heatmap
      for t, w in enumerate(W):
        figure(figsize=(10,10))
        imshow(w[:args.rank,:]); axis('image'); colorbar(); clim(0, 1)
        experiment.log_figure(figure_name='W_%s-matrix'%(t))
        close('all')

      # plot w_last heatmap
      figure(figsize=(10,10))
      imshow(self.w_last_save[:args.rank,:]); axis('image'); colorbar(); clim(0, 1)
      experiment.log_figure(figure_name='w_last-matrix')
      close('all')


      # dump W, H, T to disk
      if args.dumpdisk:
        with gzip.open(join(args.logdir, 'learned_params.joblib'), 'wb') as f:
          joblib.dump(params, f)
          experiment.log_asset(join(args.logdir, 'learned_params.joblib'))


if __name__=='__main__':

  # comet experiement
  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='ranksweep-1')
  experiment.log_parameters(vars(args))

  # front matter
  home = os.environ['HOME']
  autoname = 'rank_%s/lr_%s' % (args.rank, args.lrnrate)
  experiment.set_name(autoname)
  args.logdir = join(home, 'ckpt', autoname)
  os.makedirs(args.logdir, exist_ok=True)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  # load data from file
  maybe_download('https://www.dropbox.com/s/lu38zp3ixjpth9e/graph_data_cube.pkl?dl=0',
                 'graph_data_cube.pkl', join(home, 'datasets'), filetype='file')
  with gzip.open(join(home, 'datasets', 'graph_data_cube.pkl'), 'rb') as f:
    datacube = pickle.load(f)
  ntime, nnode, nfeat = datacube.shape

  # build tf graph
  model = Model(args, ntime, nnode, nfeat)

  # run optimizer on training data
  model.fit(datacube)

  # plot visualizations
  model.plot(ending=True)
