from comet_ml import Experiment
experiment = Experiment(api_key="vPCPPZrcrUBitgoQkvzxdsh9k", parse_args=False, project_name='nmf')
import tensorflow as tf
import numpy as np
import argparse
import os
from os.path import join, basename, dirname, exists

class Model():

  def __init__(self, args):

    self.args = args
    self.build_graph()
    self.build_sess()

  def build_graph(self):

    with tf.name_scope('inputs'):
      self.timeidx = tf.placeholder(tf.int32, [None], name='timeidx')
      self.nodeidx = tf.placeholder(tf.int32, [None], name='nodeidx')
      self.featidx = tf.placeholder(tf.int32, [None], name='featidx')
      self.labels = tf.placeholder(tf.float32, [None], name='labels')
    with tf.name_scope('hyperparams'):
      self.lr = tf.placeholder(tf.float32, [], name='lr')
      self.wdeccoef = tf.placeholder(tf.float32, [], name='wdeccoef')

    with tf.name_scope('forward'):

      W = tf.Variable(tf.truncated_normal([nnode, rank], stddev=0.2, mean=0), name='W_0')
      H = tf.Variable(tf.truncated_normal([rank, nfeat], stddev=0.2, mean=0), name='H')
      Tres = tf.Variable(tf.truncated_normal([rank, rank], stddev=0.2, mean=0), name='Tres')

      X = []
      for t in range(ntime):
        X.append(tf.matmul(W, H, name='X_'+str(t)))
        W = tf.add(W, tf.matmul(W, Tres, name='Wres_'+str(t+1)), name='W_'+str(t+1))

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
      self.trainop = tf.train.AdamOptimizer(self.lr).minimize(self.loss, name='trainop')

    self.step = tf.train.get_or_create_global_step()

    tf.summary.scalar('crit', self.crit)
    tf.summary.scalar('loss', self.loss)
    self.merged = tf.summary.merge_all()

  def build_sess(self):

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
    self.sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(self.args.logdir, graph=self.sess.graph)

  def fit(self):

    # toy data
    timeidx = np.array([0,0,1,2])
    nodeidx = np.array([0,0,1,2])
    featidx = np.array([0,1,0,1])
    labels = np.array([1.1, 0.5, 3.3, 0.2])
    lr = .1
    wdeccoef = 1e-4

    _, merged, step = self.sess.run([model.trainop, model.merged, model.step],
                       {model.timeidx: timeidx,
                        model.nodeidx: nodeidx,
                        model.featidx: featidx,
                        model.labels: labels,
                        model.lr: lr,
                        model.wdeccoef: wdeccoef,
                        })

    writer.add_summary(merged, step)

if __name__=='__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-name', default='debug', type=str)
  args = parser.parse_args()

  home = os.environ['HOME']
  args.logdir = join(home, 'ckpt', args.name)
  os.makedirs(args.logdir, exist_ok=True)

  ntime = 4
  nnode = 10
  nfeat = 5
  rank = 3

  model = Model(args)


