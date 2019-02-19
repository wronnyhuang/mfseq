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

class Model():

  def __init__(self, args):
    '''constructor of the model. create computation graph and build the session'''

    self.args = args
    self.build_graph()
    self.build_sess()

  def build_graph(self):
    '''build computational graph'''

    with tf.name_scope('hyperparams'):
      self.lr = tf.placeholder(tf.float32, [], name='lr')
      self.trancoef = tf.placeholder(tf.float32, [], name='trancoef')
      self.wdeccoef = tf.placeholder(tf.float32, [], name='wdeccoef')
      self.nnegcoef = tf.placeholder(tf.float32, [], name='nnegcoef')

    with tf.name_scope('inputs'):
      self.timeidx = tf.placeholder(tf.int32, [None], name='timeidx')
      self.nodeidx = tf.placeholder(tf.int32, [None], name='nodeidx')
      self.featidx = tf.placeholder(tf.int32, [None], name='featidx')
      self.labels = tf.placeholder(tf.float32, [None], name='labels')

    with tf.name_scope('forward'):

      self.H = tf.Variable(tf.random_gamma(shape=[args.rank, nfeat], alpha=1.0), name='H')
      self.T = tf.Variable(tf.random_gamma(shape=[args.rank, args.rank], alpha=1.0), name='T')

      X = []
      self.W = []
      self.W_tran = []
      for t in range(ntime):
        w = tf.Variable(tf.random_gamma(shape=[nnode, args.rank], alpha=1.0), name='W_'+str(t))
        X.append(tf.matmul(w, self.H, name='X_'+str(t)))
        self.W.append(w)
        if t > 0: # add transition regularizer
          w_tran = tf.matmul(self.W[-2], self.T, name='W_tran_'+str(t))
          self.W_tran.append(w_tran)

    with tf.name_scope('predictions'):
      X = tf.stack(X, axis=0, name='X')
      Xflat = tf.reshape(X, [-1], name='Xflat')
      with tf.name_scope('indices_of_datapoints'):
        indices = self.timeidx*nnode*nfeat + self.nodeidx*nfeat + self.featidx
      self.preds = tf.gather(Xflat, indices, name='Xgather')

    with tf.name_scope('loss'):
      criterion = tf.losses.mean_squared_error
      with tf.name_scope('criterion'):
        self.crit = criterion(self.preds, self.labels)
      with tf.name_scope('transition'):
        self.tran = tf.add_n([criterion(w_tran, w) for w_tran, w in zip(self.W_tran, self.W[1:])]) / len(self.W_tran)
      with tf.name_scope('weight_decay'):
        self.wdec = tf.global_norm(tf.trainable_variables())**2
      with tf.name_scope('nonneg'):
        self.nneg = tf.global_norm([tf.nn.relu(-t) for t in tf.trainable_variables()])**2

      self.loss = tf.add_n([self.crit, self.trancoef * self.tran, self.wdeccoef * self.wdec, self.nnegcoef * self.nneg], name='loss')
      self.step = tf.train.get_or_create_global_step()
      self.trainop = tf.train.AdamOptimizer(self.lr).minimize(self.loss, name='trainop', global_step=self.step)


    tf.summary.scalar('train/crit', self.crit)
    tf.summary.scalar('train/loss', self.loss)
    tf.summary.scalar('lr', self.lr)
    self.merged = tf.summary.merge_all()

  def build_sess(self):
    '''start tf session'''

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())
    self.writer = tf.summary.FileWriter(self.args.logdir, graph=self.sess.graph)

  def test(self, testloader, step):
    '''test on entire test set'''

    # test over all test data
    running_crit = 0
    for i, batch in enumerate(testloader):
      timeidx, nodeidx, featidx, labels = batch.numpy().transpose()
      crit, = self.sess.run([self.crit],
                            {self.timeidx: timeidx,
                             self.nodeidx: nodeidx,
                             self.featidx: featidx,
                             self.labels: labels,
                             })
      running_crit += crit * len(batch)
    avg_crit = running_crit / len(testloader.dataset)
    experiment.log_metric('test/crit', avg_crit, step)
    print('TEST:\tstep', step, '\tcrit', avg_crit)


  def fit(self, trainloader, testloader):
    '''fit the model to the data presented in the input dataloader'''

    step = 0
    self.metrics = dict(loss=self.loss, crit=self.crit, tran=self.tran, wdec=self.wdec, nneg=self.nneg)
    for epoch in range(args.nepoc):

      # train for an epoch
      for i, batch in enumerate(trainloader):

        dropfactor = 10 if step > args.lrdrop else 1
        timeidx, nodeidx, featidx, labels = batch.numpy().transpose()
        _, metrics, step = self.sess.run([self.trainop, self.metrics, self.step],
                           {self.timeidx: timeidx,
                            self.nodeidx: nodeidx,
                            self.featidx: featidx,
                            self.labels: labels,
                            self.lr: args.lrnrate / dropfactor,
                            self.trancoef: args.trancoef,
                            self.wdeccoef: args.wdeccoef,
                            self.nnegcoef: args.nnegcoef,
                            })
        if np.mod(step, args.logtrain)==0:
          experiment.log_metrics(metrics, step=step)
          print('TRAIN:\tepoch', epoch, '\tstep', step, '\tcrit', metrics['crit'], '\tloss', metrics['loss'])
        if np.mod(step, args.logtest)==0:
          self.test(testloader, step)
          if args.logimage: self.plot()

    print('done training')

  def get_params(self):
    return self.sess.run(dict(W=self.W, H=self.H, T=self.T))

  def plot(self, ending=False):
    '''plot and save distributions'''

    # distribution of parameters
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
      hist(labels, 200); xlim(-.1, 2); title('distribution-labels')
      experiment.log_figure(figure=plt.gcf())

      # dump W, H, T to disk
      with open(join(args.logdir, 'learned_params.joblib'), 'wb') as f:
        joblib.dump(params, f)
        experiment.log_asset(join(args.logdir, 'learned_params.joblib'))

class GraphDataset(torch.utils.data.Dataset):
  '''dataset object for the aml graph data with features extracted via refex'''

  def __init__(self, timeidx, nodeidx, featidx, labels, train=True):

    self.data = np.array(list((zip(timeidx, nodeidx, featidx, labels))))
    condition = timeidx != timeidx.max() if train else timeidx == timeidx.max()
    self.data = self.data[condition]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


if __name__=='__main__':

  # parse terminal arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-name', default='debug', type=str)
  parser.add_argument('-rank', default=10, type=int)
  parser.add_argument('-batchsize', default=20000, type=int)
  parser.add_argument('-lrnrate', default=.1, type=float)
  parser.add_argument('-lrdrop', default=70, type=int)
  parser.add_argument('-trancoef', default=10, type=float)
  parser.add_argument('-wdeccoef', default=5e-5, type=float)
  parser.add_argument('-nnegcoef', default=1e-1, type=float)
  parser.add_argument('-nepoc', default=1, type=int)
  parser.add_argument('-logtrain', default=2, type=int)
  parser.add_argument('-logtest', default=5, type=int)
  parser.add_argument('-gpu', default='0', type=str)
  parser.add_argument('-logimage', action='store_true')
  parser.add_argument('-randname', action='store_true')
  args = parser.parse_args()

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
  maybe_download('https://www.dropbox.com/s/cdu4xx8vnavkvz3/graph_data_table.pkl?dl=0',
                 'graph_data_table.pkl', join(home, 'datasets'), filetype='file')
  with gzip.open(join(home, 'datasets', 'graph_data_table.pkl'), 'rb') as f:
    (timeidx, nodeidx, featidx), labels, (ntime, nnode, nfeat) = pickle.load(f)

  # create dataloader object
  trainloader = torch.utils.data.DataLoader(GraphDataset(timeidx, nodeidx, featidx, labels, True), batch_size=args.batchsize, shuffle=True, num_workers=0)
  testloader = torch.utils.data.DataLoader(GraphDataset(timeidx, nodeidx, featidx, labels, False), batch_size=args.batchsize, shuffle=False, num_workers=0)

  # build tf graph
  model = Model(args)

  # run optimizer on training data
  model.fit(trainloader, testloader)

  # plot stuff
  model.plot(ending=True)
