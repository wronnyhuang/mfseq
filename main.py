# from comet_ml import Experiment
# experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='nmf')
import tensorflow as tf
import numpy as np
import argparse
import os
from os.path import join, basename, dirname, exists
import pickle
import gzip
import torch.utils.data
from time import time
from utils import maybe_download

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
      self.wdeccoef = tf.placeholder(tf.float32, [], name='wdeccoef')

    with tf.name_scope('inputs'):
      self.timeidx = tf.placeholder(tf.int32, [None], name='timeidx')
      self.nodeidx = tf.placeholder(tf.int32, [None], name='nodeidx')
      self.featidx = tf.placeholder(tf.int32, [None], name='featidx')
      self.labels = tf.placeholder(tf.float32, [None], name='labels')

    with tf.name_scope('forward'):

      # W = tf.Variable(tf.truncated_normal([nnode, args.rank], stddev=0.2, mean=0), name='W_0')
      H = tf.Variable(tf.truncated_normal([args.rank, nfeat], stddev=0.2, mean=0), name='H')
      Tres = tf.Variable(tf.truncated_normal([args.rank, args.rank], stddev=0.2, mean=0), name='Tres')

      X = []
      W = []
      for t in range(ntime):
        w = tf.Variable(tf.truncated_normal([nnode, args.rank], stddev=0.2, mean=0), name='W_'+str(t))
        X.append(tf.matmul(w, H, name='X_'+str(t)))
        W.append(w)
        # W = tf.add(W, tf.matmul(W, Tres, name='Wres_'+str(t+1)), name='W_'+str(t+1))

    with tf.name_scope('predictions'):
      X = tf.stack(X, axis=0, name='X')
      Xflat = tf.reshape(X, [-1], name='Xflat')
      with tf.name_scope('indices_of_datapoints'):
        indices = self.timeidx*nnode*nfeat + self.nodeidx*nfeat + self.featidx
      self.preds = tf.gather(Xflat, indices, name='Xgather')

    with tf.name_scope('weight_decay'):
      self.wdec = self.wdeccoef * tf.global_norm(tf.trainable_variables())

    with tf.name_scope('loss'):

      with tf.name_scope('criterion'):
        criterion = lambda preds, labels: tf.reduce_mean( ( preds - labels )**2 )
        self.crit = criterion(self.preds, self.labels)

      self.loss = tf.add(self.crit, self.wdec, name='loss')
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

  def fit(self, trainloader, testloader):
    '''fit the model to the data presented in the input dataloader'''

    for epoch in range(10):

      # train for an epoch
      elapsed = 0
      for i, batch in enumerate(trainloader):

        start = time()
        timeidx, nodeidx, featidx, labels = batch.numpy().transpose()
        _, merged, loss, crit, step = self.sess.run([self.trainop, self.merged, self.loss, self.crit, self.step],
                           {self.timeidx: timeidx,
                            self.nodeidx: nodeidx,
                            self.featidx: featidx,
                            self.labels: labels,
                            self.lr: args.lr,
                            self.wdeccoef: args.wdeccoef,
                            })
        elapsed += time()-start
        logperiod = 1
        if np.mod(step, logperiod) == 0:
          self.writer.add_summary(merged, step)
          print('TRAIN: epoch', epoch, '\tstep', step, '\tcrit', crit, '\tloss', loss, '\telapsed', elapsed/logperiod)
          elapsed = 0

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
      self.log_avgs_hack(avg_crit)
      print('TEST: epoch', epoch, '\tstep', step, '\tcrit', crit)

  def log_avgs_hack(self, avg_crit):

    if self.merged_test not in locals():
      self.testcrit = tf.get_variable(name='testcrit')
      summary_crit = tf.summary.scalar('test/crit', testcrit, step)
      self.merged_test = tf.summary.merge(summary_crit)
    merged = self.sess.run([self.merged_test], {self.testcrit: avg_crit})
    self.writer.add_summary(merged, step)


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
  parser.add_argument('-rank', default=3, type=int)
  parser.add_argument('-batchsize', default=100000, type=int)
  parser.add_argument('-lr', default=.1, type=float)
  parser.add_argument('-wdeccoef', default=1e-4, type=float)
  args = parser.parse_args()

  # front matter
  home = os.environ['HOME']
  args.logdir = join(home, 'ckpt', args.name)
  os.makedirs(args.logdir, exist_ok=True)

  # load data from file
  maybe_download('https://www.dropbox.com/s/pc8qggvvm2gfebl/graph_data_table.pkl?dl=0',
                 'graph_data_table.pkl', os.getcwd(), filetype='file')
  with gzip.open('graph_data_table.pkl', 'rb') as f:
    (timeidx, nodeidx, featidx), labels, (ntime, nnode, nfeat) = pickle.load(f)

  # create dataloader object
  trainloader = torch.utils.data.DataLoader(GraphDataset(timeidx, nodeidx, featidx, labels, True), batch_size=args.batchsize, shuffle=True, num_workers=0)
  testloader = torch.utils.data.DataLoader(GraphDataset(timeidx, nodeidx, featidx, labels, False), batch_size=args.batchsize, shuffle=False, num_workers=0)

  # build tf graph
  model = Model(args)

  # run optimizer on training data
  model.fit(trainloader, testloader)



