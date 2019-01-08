#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import argparse
import os
import pickle
import time

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from seq2seq_model_v2 import vision, get_optimizer, Model
from helper import add_left_context
# import matplotlib.pyplot as plt
import matplotlib as mpl
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'axes.grid': True,
   'grid.linestyle': 'dotted',
   'grid.linewidth': 0.99,
   'text.usetex': False,
   'figure.figsize': [5,5]
   }
mpl.rcParams.update(params)
import matplotlib.pyplot as plt
from stGraph import ST_GRAPH
from tensorflow.contrib import keras
# import seaborn as sns; sns.set(color_codes=True);

# train on only one gpu
# os.environ["CUDA_VISIBLE_DEVICES"]="%s" % args.gpu
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

iterator = 1
loss_graph = []
global lr

def Train(args):
    global loss_graph
    global lr
    lr = 1e-3
    print("\nRead graph from file %s " % args.stgraph)
    g = pickle.load( open(args.stgraph, 'rb') )
    print("\nModel initializing...")
    model = Model(args)
    print("\nDone. Start tensorflow session.")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        generator = give_me_batch(args, g)
        best_loss = None
        for epoch in range(1, args.num_epochs):
            if epoch % 25 == 0:
                lr *= .5
            loss = RunEpoch(epoch, args, sess, generator, model, g)
            if best_loss is None or best_loss > loss:
                checkpoint_path = os.path.join(args.ckpt_dir, 'model_31.ckpt')
                saver.save(sess, checkpoint_path, global_step=480*10)
                print("model saved to {}".format(checkpoint_path))
                pickle.dump(loss_graph, open('./loss_graph_33.p', 'wb'))
                best_loss = loss

        
def RunEpoch(epoch, args, sess, generator, model, graph):
    global loss_graph
    global lr
    mean = []
    feature_extractor = keras.applications.VGG16(include_top=False,
                                                     weights="imagenet")
    # lr_decayed = args.learning_rate * args.decay_rate ** epoch
    lr_decayed = lr
    for i in range(1, int(graph.len / args.batch_size)):        
        fs, es, ns = next(generator)
        # input_files = np.array(fs); input_files = input_files[:,:args.seq_len]
        # input_files = fs[:][:args.seq_len]
        input_edges = es[:, :args.seq_len, :]
        input_target = es[:, args.seq_len:, :]
        # assert input_files.shape == (20,15), print('DEBUG STATEMENT')
        # add left context in order to apply spatio-temporal conv
        input_files = add_left_context(fs, 5, args.batch_size)
        input_files = np.array(fs)[:,:args.seq_len+args.left_context]
        
        state = sess.run(model.initial_state)
        feed_dict = ({model.lr : lr_decayed,
                      model.input_data : input_edges,
                      model.target_data : input_target,
                      model.lr : lr_decayed,
                      model.initial_state : state,
                      model.input_files : input_files
        })
        try:
            train_loss, state, lrate, _= sess.run([model.cost,
                                                   model.final_state, model.lr,
                                                   model.train_op],
                                                  feed_dict=feed_dict)
        except Exception as e:
            print(e)
            continue 
        mean.append(train_loss)
        loss_graph.append(train_loss)
        print('\rStep: %d' % i,
              'Train loss: %.4f' % train_loss,
              'μ loss %.4f' % (sum(mean) / i), '--', lrate, end= ' ')
    print('start epoch: %d' % (epoch + 1))
    return sum(mean) / i

def Test(args):
    print("\nRead graph from file %s " % args.stgraph)
    g = pickle.load( open(args.stgraph, 'rb') )

    model = Model(args, True)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    # define generator
    generator = give_me_batch(args, g)
    fs, es, ns = next(generator)

    # format data
    input_files  = fs[:][:15] #4x24, 4X12
    input_edges  = es[:, :15, :]
    input_target = es[:, 15:, :]
    input_nodes  = ns[:, :15, :]

    # add left context in order to apply spatio-temporal conv
    input_files = add_left_context(fs, 5, args.batch_size)
    input_fiels = input_files[0][:args.seq_len]
    input_files = np.array(input_files)# [0,:args.seq_len+args.left_context]

    # frames = input_files[0].reshape(1, -1)
    # print(frames.shape, 'shape of frames')
    obs_traj = input_edges[0]
    final_traj = input_target[0]

    # call the police!
    complete_traj = model.sample(sess=sess,
                                 traj=obs_traj,
                                 future_traj=final_traj,
                                 frames=input_files,
                                 nodes_in=input_nodes,
                                 num=15)

    # xmin, xmax, ymin, ymax = -24.980316, 24.92566, -24.895721, 24.967285
    xmin, xmax, ymin, ymax = -19.91409, 24.549171, 1.0074499, 31.188437
    print('asd asd asd ')
    print(complete_traj.shape)
    # unscale data 
    x = final_traj
    x2 = complete_traj[15:,:]
    mean, std = 9.574188, 9.673232

    x[:,0] = (x[:,0] + 1) / 2 * (xmax - xmin) + xmin
    x[:,1] = (x[:,1] + 1) / 2 * (ymax - ymin) + ymin

    x2[:,0] = (x2[:,0] + 1) / 2 * (xmax - xmin) + xmin
    x2[:,1] = (x2[:,1] + 1) / 2 * (ymax - ymin) + ymin

    x = (x * std) + mean
    x2 = (x2 * std) + mean

    print('DEBUG')
    print(x2.shape)
    print('DEBUG -- 2')
    print(x.shape)
    # radar = np.array([100, 100], dtype=np.int8)
    images = []
    test_images = np.array(fs)[:,15:]
    """
    for iter, file in enumerate(test_images[0]):
        radar = np.zeros([100, 100])
        radar[:,:] = 0
        print(file)
        img = cv2.imread(file)
        xs = int(50 + x[iter,0])
        ys = int(50 + x[iter, 1])
        xs2 = int(50 + x2[iter, 0])
        ys2 = int(50 + x2[iter, 1])
        print(xs, ys)
        cv2.line(radar, (50,50), (xs, ys), (255, 255, 255), 1)
        cv2.line(radar, (50, 50), (xs2, ys2), (255, 0, 0), 1)
        # vis = np.concatenate((img, radar), axis=0)
        # cv2.imshow('frame', vis)
        # images.append(vis)
        cv2.imshow('frame', img)
        cv2.imshow('frame2', radar)
        if cv2.waitKey(700) == ord('q'):
            cv2.destroyAllWindows()
        # print(iter)
    """

    # cv2.destroyAllWindows()


    x = x*.8
    x2 = x2*.8

    plt.subplot(211)
    plt.plot(x[:,0], '-+k', label='X ground truth in m')
    plt.plot(x2[:,0], '->r', label='X predicted in m')
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.subplot(212)
    plt.plot(x[:,1], '-+k', label='Y ground truth in m')
    plt.plot(x2[:,1], '->r', label='Y predicted in m')
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.show()
    # calculate euclidian distance btw 2 vectors
    error = np.linalg.norm(x - x2) * 0.8

    # save images into .GIF
    # import imageio
    # imageio.mimsave('./test.gif', images)

    print('Displacement error: ~%.4f meters' % error)


def SaveModel(args, sess):
    pass

from sklearn.utils import shuffle
def give_me_batch(args, g):
    filenames = []
    edges = []
    nodes = []
    for i in range(g.len):
        fs, ns, es, es_t = g.process_seq(g.sequences[i])
        if np.min(es[:,:,1]) <= 1: continue
        # print(i)
        if ns.shape[0] == 1:
            # print(i)
            filenames.append(fs)
            edges.append(es.reshape(-1, 2))
            nodes.append(ns.reshape(-1, 2))

    XS, YS, NS = shuffle(edges, filenames, nodes)
    XS = np.array(XS, dtype=np.float32)
    NS = np.array(NS, dtype=np.float32)

    # normalize data between [-1, 1]
    xmin, xmax = np.min(XS[:,:,0]), np.max(XS[:,:,0])
    ymin, ymax = np.min(XS[:,:,1]), np.max(XS[:,:,1])
    # mean, std = np.mean(XS), np.std(XS)
    mean, std = np.mean(XS), np.std(XS)
    print(mean, std)
    XS = (XS - mean) / std

    XS[:,:,0] = -1 + 2.0 * (XS[:,:,0] - xmin) / (xmax - xmin)
    XS[:,:,1] = -1 + 2.0 * (XS[:,:,1] - ymin) / (ymax - ymin)
    # mean, std = np.mean(XS), np.std(XS)
    # XS = (XS - mean) / std
    print(xmin, xmax, ymin, ymax)

    xmin, xmax = np.min(NS[:,:,0]), np.max(NS[:,:,0])
    ymin, ymax = np.min(NS[:,:,1]), np.max(NS[:,:,1])
    NS[:,:,0] = 2 * (NS[:,:,0] - xmin) / (xmax - xmin) - 1
    NS[:,:,1] = 2 * (NS[:,:,1] - ymin) / (ymax - ymin) - 1
    print(xmin, xmax, ymin, ymax)

    start, end = 0, args.batch_size
    while 1:
        for i in range(0, int(XS.shape[0] / args.batch_size)):
            # if end, re-count pointers
            if i == int(XS.shape[0] / args.batch_size) - 1:
                i = 0
                start, end = 0, args.batch_size
            e = XS[start:end]
            f = YS[start:end]
            n = NS[start:end]
            start = end
            end += args.batch_size
            yield f, e, n

# TODO: move it all to the helper file
# mean error metric to analyse final trajectories
def get_mean_error(predicted_traj, true_traj, observed_length):
    error = np.zeros(len(true_traj) - observed_length)
    for i in range(observed_length, len(true_traj)):
        pred_pos = predicted_traj[i, :]
        true_pos = true_traj[i, :]
        error[i-observed_length] = np.linalg.norm(true_pos - pred_pos)
    return np.mean(error)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stgraph', default='new_with_disp.p',
                        help='processed and serialized graph')
    parser.add_argument('--df',
                        default='/home/user/src/driver_behavior/dataset.py',
                        help='dataframe to be processed')
    parser.add_argument('--node_rnn_size', type=int, default=128,
                        help='define node RNN size')
    parser.add_argument('--cell', default='lstm',
                        help='rnn cell type, can be \lstm or \gru')

    # RNN SIZE HYPER-PARAMETER
    parser.add_argument('--rnn_size', type=int, default=32,
                        help='define basic RNN size')
    parser.add_argument('--rnn_proj', type=int, default=32,
                        help='define RNN proection dim')
    parser.add_argument('--edge_rnn_size', type=int, default=128,
                        help='define edge RNN size')
    parser.add_argument('--node_dim', type=int, default=2,
                        help='node input dimension')
    parser.add_argument('--edge_dim', type=int, default=2,
                        help='edge input dimension')
    parser.add_argument('--video_height', type=int, default=160,
                        help='input image height')
    parser.add_argument('--video_width', type=int, default=320,
                        help='input image weight')
    parser.add_argument('--left_context', type=int,  default=5,
                        help='size of left context')

    # EMBED SIZE HYPER-PARAMETER
    parser.add_argument('--embed_size', type=int, default=128,
                        help='node embeded layer size')
    parser.add_argument('--attention_size', type=int, default=128,
                        help='attention size')

    # SEQ LEN HYPER-PARAMETER
    parser.add_argument('--seq_len', type=int, default=15,
                        help='length of sequence')
    parser.add_argument('--push_len', type=int, default=15,
                        help='length of sequence to predict trajectory')
    parser.add_argument('--pred_len', type=int, default=15,
                        help='length of predicted trajectory')

    # BATCH SIZE HYPER-PARAMETER
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--epoch_num', type=int, default=500,
                        help='numer of epoch')
    parser.add_argument('--gradient_clip_size', type=int, default=5.,
                        help='gradient clipping range')
    parser.add_argument('--l2_reg', type=float, default=1e-8,
                        help='l2 regularization hyper parameter')

    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate hyperparameter')
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for rmsprop')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout rate hyperparameter')
    parser.add_argument('--log_dir', default='./log',
                        help='log directory, place to store training info')
    parser.add_argument('--mode', default='train',
                        help='train, valid or test modes')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='total number of epochs')
    parser.add_argument('--train', default=True, help='Train or Inference?')
    parser.add_argument('--gpu', default=1, help='Train on wich GPU? <1,2>')
    parser.add_argument('--ckpt_dir', default='./lstm_updated',
                        help='checkpoint saved path')
    parser.add_argument('--output_dim', default=5, help='Network output dim.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="%s" % args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
    
    print('### Train ###') if args.train is True else print('### Test ###')
    Train(args) if args.train is True else Test(args)

if __name__ == '__main__':
    main()
