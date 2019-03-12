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
from matplotlib.pyplot import plot, imshow, colorbar, show, axis, hist, subplot, xlabel, ylabel, title, legend, savefig, figure, close, suptitle, tight_layout, xlim, ylim
import matplotlib.pyplot as plt

# parse terminal arguments
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-name', default='debug', type=str)
parser.add_argument('-rank', default=10, type=int)
parser.add_argument('-batchsize', default=20000, type=int)
parser.add_argument('-lrnrate', default=.1, type=float)
parser.add_argument('-lrdrop', default=70, type=int)
parser.add_argument('-trancoef', default=.1, type=float)
parser.add_argument('-wdeccoef', default=5e-3, type=float)
parser.add_argument('-nnegcoef', default=1e-1, type=float)
parser.add_argument('-maxgradnorm', default=10, type=float)
parser.add_argument('-nepoc', default=100, type=int)
parser.add_argument('-logtrain', default=2, type=int)
parser.add_argument('-logtest', default=5, type=int)
parser.add_argument('-logimage', action='store_true')
parser.add_argument('-randname', action='store_true')
args = parser.parse_args()

class Model():

  def __init__(self, args):
    '''constructor of the model. create computation graph and build the session'''

    self.args = args
    self.build_graph()
    self.build_sess()


  def build_graph(self):
    '''build computational graph'''

    with tf.name_scope('inputs'):
      self.lr = tf.placeholder(tf.float32, [], name='lr')
      self.X_true = tf.placeholder(tf.float32, datacube.shape, name='X_true')
      self.gategrad = tf.placeholder(tf.float32, datacube.shape, name='gategrad')

    with tf.name_scope('forward'):
      self.H = tf.Variable(tf.random_gamma(shape=[args.rank, nfeat], alpha=1.0), name='H', trainable=False)
      self.T = tf.Variable(tf.random_gamma(shape=[args.rank, args.rank], alpha=1.0), name='T', trainable=False)
      self.W = []
      X_pred = []
      X_tran = []

      for t in range(ntime):
        w = tf.Variable(tf.random_gamma(shape=[nnode, args.rank], alpha=1.0), name='W_'+str(t))
        X_pred.append(tf.matmul(w, self.H, name='X_pred_'+str(t)))
        self.W.append(w)
        if t >= 2: # add transition regularizer
          x_tran = tf.add(tf.matmul(tf.matmul(self.W[-3], self.W[-3]), self.T), tf.matmul(self.W[-2], self.T), name='X_tran_'+str(t))
          X_tran.append(x_tran)
      X_pred = tf.stack(X_pred, axis=0, name='X_pred')

    criterion = tf.losses.mean_squared_error

    with tf.name_scope('regularizers'):
      with tf.name_scope('transition'):
        self.tran = tf.add_n([criterion(x_true, x_pred) for x_true, x_pred in zip(self.X_true[2:], self.X_tran]) / len(self.X_tran)
      with tf.name_scope('weight_decay'):
        self.wdec = tf.global_norm(tf.trainable_variables())
      with tf.name_scope('nonneg'):
        self.nneg = tf.global_norm([tf.nn.relu(-t) for t in tf.trainable_variables()])**2

    with tf.name_scope('loss'):
      self.crit = criterion(X_pred, self.X_true)
      self.loss = tf.add_n([(1-args.trancoef) * self.crit, args.trancoef * self.tran, args.wdeccoef * self.wdec, args.nnegcoef * self.nneg], name='loss')

    with tf.name_scope('train_ops'):
      # keep track of training step
      self.step = tf.train.get_or_create_global_step()
      # clip gradients by a max norm value
      opt = tf.train.AdamOptimizer(self.lr)
      grads = tf.gradients(self.loss, tf.trainable_variables())
      grads, self.gradnorm = tf.clip_by_global_norm(grads, args.maxgradnorm)
      # training op
      self.trainop = opt.apply_gradients(zip(grads, tf.trainable_variables()), global_step=self.step, name='trainop')

    # # tensorboard summaries
    # tf.summary.scalar('train/crit', self.crit)
    # tf.summary.scalar('train/loss', self.loss)
    # tf.summary.scalar('lr', self.lr)
    # self.merged = tf.summary.merge_all()


  def build_sess(self):
    '''start tf session'''

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(self.args.logdir, graph=self.sess.graph)


  def test(self, step):
    '''test on entire test set'''
    crit = self.sess.run(self.crit, {self.X_true: datacube})
    experiment.log_metric('test/crit', crit, step)
    print('TEST:\tstep', step, '\tcrit', crit)


  def fit(self):
    '''fit the model to the data presented in the input dataloader'''
    step = 0
    self.metrics = dict(loss=self.loss, crit=self.crit, tran=self.tran, wdec=self.wdec, nneg=self.nneg, gradnorm=self.gradnorm)
    for epoch in range(args.nepoc):
      dropfactor = 0.1 if step > args.lrdrop else 1
      _, metrics, step, = self.sess.run([self.trainop, self.metrics, self.step],
                                        {self.X_true: datacube,
                                         self.lr: args.lrnrate * dropfactor,
                                         self.gategrad: gate,
                                        })
      if np.mod(step, args.logtrain)==0:
        experiment.log_metrics(metrics, step=step)
        print('TRAIN:\tepoch', epoch, '\tstep', step, '\tcrit', metrics['crit'], '\tloss', metrics['loss'])
      if np.mod(step, args.logtest)==0:
        self.test(step)
        if args.logimage: self.plot()
    print('done training')


  def get_params(self):
    return self.sess.run(dict(W=self.W, H=self.H, T=self.T))


  def plot(self, ending=False):
    '''plot and save distributions'''
    params = self.get_params()
    T, H, W = params['T'], params['H'], params['W']
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
    hist(T.ravel(), 10); xlim(-.1, 2); title(figname)
    experiment.log_figure(figure_name=figname, figure=plt.gcf())
    close('all')

    if ending:

      # plot distribution of labels
      hist(datacube.ravel()[np.abs(datacube.ravel())>1e-2].ravel(), 200); xlim(-.1, 2); title('distribution-labels')
      experiment.log_figure(figure=plt.gcf())

      # dump W, H, T to disk
      with open(join(args.logdir, 'learned_params.joblib'), 'wb') as f:
        joblib.dump(params, f)
        experiment.log_asset(join(args.logdir, 'learned_params.joblib'))


if __name__=='__main__':

  # comet experiement
  experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='nmf-ranksweep')
  experiment.log_parameters(vars(args))

  # front matter
  home = os.environ['HOME']
  autoname = 'rank_%s/lr_%s/wdeccoef_%s' % (args.rank, args.lrnrate, args.wdeccoef)
  experiment.set_name(autoname)
  args.name = autoname
  args.logdir = join(home, 'ckpt', args.name)
  os.makedirs(args.logdir, exist_ok=True)
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  # load data from file
  maybe_download('https://www.dropbox.com/s/lu38zp3ixjpth9e/graph_data_cube.pkl?dl=0',
                 'graph_data_cube.pkl', join(home, 'datasets'), filetype='file')
  with gzip.open(join(home, 'datasets', 'graph_data_cube.pkl'), 'rb') as f:
    datacube = pickle.load(f)

  ntime, nnode, nfeat = datacube.shape
  gate = np.ones([ntime, nnode, args.rank]) # gate is multiplied against gradients wrt W
  nodeidx = np.random.permutation(nnode)
  split = int(.9*len(nodeidx))
  gate[:, nodeidx[split:], :] = 0

  # build tf graph
  model = Model(args)

  # run optimizer on training data
  model.fit()

  # plot stuff
  model.plot(ending=True)
