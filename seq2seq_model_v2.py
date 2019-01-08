
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
import tensorflow as tf
import numpy as np
from vision import apply_3d_vision
slim = tf.contrib.slim
LSTM = tf.contrib.rnn.LSTMCell
seq2seq = tf.contrib.legacy_seq2seq.rnn_decoder
keras = tf.contrib.keras

# simple layer norm lambda func
layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x,
                                                    center=True,
                                                    scale=True,
                                                    activation_fn=None,
                                                    trainable=True,
                                                    )
xavier = tf.contrib.layers.xavier_initializer()

# clip gradient by their global_norm value
def get_optimizer(loss, l_rate=1e-4, clip_value=10, moment=.9):
    optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=l_rate, momentum=moment)
    gradvars = optimizer.compute_gradients(loss)
    gradients, v = zip(*gradvars)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_value)
    return optimizer.apply_gradients(zip(gradients, v))

# linear layer: x.some_shape to x.needed_shape
def linear_projec(x,
                 embed_size,
                 var_name,
                 dropout=None,
                 keep_prob=None,
                 activation=None,
                 scope=None,
                 reuse=None,
):
    with tf.variable_scope(scope, var_name, [x], reuse=reuse):
        res = tf.contrib.layers.linear(
            x, embed_size, activation_fn=None if not activation else activation)
        if dropout:
            res = tf.nn.dropout(res, keep_prob=keep_prob)
    return res

# vision module,process batch of sequence of frames
def vision(video,
           bs,
           seq_len,
           img_w,
           img_h,
           img_c,
           feature_len,
           keep_prob,
           scope=None,
           reuse=None,
):
    video = tf.reshape(video, shape=[bs, seq_len, img_h, img_w, img_c])
    with tf.variable_scope(scope, 'Video', [video], reuse=reuse):
        # first res block
        net = slim.convolution2d(video, 
                                 num_outputs=64, 
                                 kernel_size=[1, 12, 12], 
                                 stride=[1, 6, 6], 
                                 padding='VALID',
                                )
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux1 = slim.fully_connected(tf.reshape(net, [bs, seq_len, -1]), 
                                    feature_len, activation_fn=None, 
                                    scope='aux_1')
        # second res block
        net = slim.convolution2d(net,
                                num_outputs=64,
                                kernel_size=[1, 6, 6],
                                stride=[1, 3, 3],
                                padding='VALID')
        net = slim.nn.dropout(x=net, keep_prob=keep_prob)
        aux2 = slim.fully_connected(tf.reshape(net, [bs, seq_len, -1]),
                                    feature_len, activation_fn=None)
        # third res block
        net = slim.convolution2d(net, 
                                num_outputs=64,
                                kernel_size=[1, 3, 3],
                                stride=[1, 1, 1],
                                padding='VALID')
        net = slim.nn.dropout(x=net, keep_prob=keep_prob)
        aux3 = slim.fully_connected(tf.reshape(net, [bs, seq_len, -1]),
                                   feature_len, activation_fn=None)
        # fourth res block
        net = slim.convolution2d(net, 
                                num_outputs=64,
                                kernel_size=[1, 3, 3],
                                stride=[1, 1, 1],
                                padding='VALID')
        net = slim.nn.dropout(x=net, keep_prob=keep_prob)
        aux4 = slim.fully_connected(tf.reshape(net, [bs, seq_len, -1]),
                                   feature_len, activation_fn=None)
        # FC layers and output
        net = slim.fully_connected(tf.reshape(net, [bs, seq_len, -1]),
                                   1024,
                                   activation_fn=None)
        net = slim.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(tf.reshape(net, [bs, seq_len, -1]),
                                   512,
                                   activation_fn=None)
        net = slim.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(tf.reshape(net, [bs, seq_len, -1]),
                                   256,
                                   activation_fn=None)
        net = slim.nn.dropout(x=net, keep_prob=keep_prob)
        net = slim.fully_connected(net, feature_len, activation_fn=None)
        net = slim.nn.dropout(x=net, keep_prob=keep_prob)
    return layer_norm(tf.nn.relu(net + aux1 + aux2 + aux3 + aux4))


class Model(object):
    """
    class to represent main model
    instead of script, can define flexible seq_len parameters, in order to use
    it like an inference model.
    """
    def __init__(self, args, infer=False):
        # if infererence, seq and bs is equal 1
        # some test experiments, just check it out
        if not infer:
            lstm_prob=tf.placeholder_with_default(input=.25, shape=())
            keep_prob=tf.placeholder_with_default(input=.25, shape=())
        else:
            args.batch_size = 1
            args.seq_len = 1
            args.pred_len = 1
            lstm_prob=tf.placeholder_with_default(input=1.0, shape=())
            keep_prob=tf.placeholder_with_default(input=1.0, shape=())
            # keep_prob=tf.placeholder_with_default(input=.25, shape=())
        self.args = args
        # define placeholders
        # video processing
        self.input_files = tf.placeholder(shape=(None,args.push_len+args.left_context),
                                          dtype=tf.string,name='filenames')
        input_images = tf.stack([tf.image.decode_jpeg(tf.read_file(x)) for x in\
                             tf.unstack(tf.reshape(self.input_files,
                                                   shape=[args.batch_size*(args.push_len+args.left_context)]
                             ))])
        input_images = -1 + 2.0 * tf.cast(input_images, tf.float32) / 255.0
        video = apply_3d_vision(args.batch_size,
                                input_images,
                                keep_prob,
                                args.seq_len,
                                args)
        self.video_reshaped = tf.nn.dropout(x=video, keep_prob=keep_prob)
        # video  = vision(input_images,
        #                 args.batch_size,
        #                 args.seq_len,
        #                 args.video_width,
        #                 args.video_height,
        #                 3,
        #                 feature_len=args.embed_size,
        #                 keep_prob=keep_prob,
        # )
        # self.video_reshaped = tf.nn.dropout(x=video, keep_prob=keep_prob)

        # edges and target processing
        self.input_data = tf.placeholder(dtype=tf.float32,
                                         shape=(None, args.push_len, 2), name='features')
        self.target_data = tf.placeholder(dtype=tf.float32,
                                          shape=(None, args.push_len, 2), name='labels')

        # define multiple lstm cells
        if args.cell is 'lstm':
            cell = tf.contrib.rnn.LSTMCell(num_units=args.rnn_size,
                                           initializer=xavier,
                                           )
        else:
            cell = tf.contrib.rnn.GRUCell(num_units=args.rnn_size)
            print('Use GRU cell instead.\n')
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=lstm_prob, seed=232)
        # cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)

        self.cell = cell
        self.initial_state = cell.zero_state(batch_size=args.batch_size,
                                             dtype=tf.float32)
        self.lr = tf.placeholder_with_default(input=1e-4, shape=())
                                     
        # 5 -> [mux, muy, sx, sy, corr]
        output_size = 5
        with tf.variable_scope("edges_embedding"):
            #  The spatial embedding using a ReLU layer
            #  Embed the 2D coordinates into embedding_size dimensions
            #  TODO: (improve) For now assume embedding_size = rnn_size
            #  CHANGE 130 TO 2 IF USE ONLY SENSORIC INPUT
            edges_emb_w = tf.get_variable("edges_embed_w",
                                          [2, args.embed_size],
                                          dtype=tf.float32,
                                          initializer=xavier
            )
            edges_emb_b = tf.get_variable("edges_embed_b", [args.embed_size],
                                          dtype=tf.float32)

        inputs = tf.split(self.input_data, args.push_len, axis=1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        # inputs = self.input_data

        # video = tf.split(self.video_reshaped, args.push_len, axis=1)
        # self.video = [tf.squeeze(video_, [1]) for video_ in video] # 12
        
        embeded = []
        target_gt = []

        # point for conv3d conv data
        self.frames = video

        # EDGES PREPROCESSING
        for iter, tensor in enumerate(inputs):
            # COMMENT THIS LINE IF USE ONLY SENSORIC INPUT
            tensor = tf.nn.relu(tf.add(tf.matmul(tensor, edges_emb_w),
                                       edges_emb_b))
            # tensor = tf.concat(values=[tensor, self.frames[iter]], axis=1)
            # tensor = tf.nn.relu(tf.add(tf.matmul(tensor, embedding_w),
            #                           embedding_b))
            embeded.append(tensor)

        # embeds = tf.reshape(tf.concat(embeded, 1), [-1, args.seq_len, args.embed_size]) # 8x20x32
        embeds = tf.reshape(tf.concat(embeded, 1), [-1, args.embed_size]) # 96x32
        embeds_norm = layer_norm(embeds)
        # visual_feeds = tf.reshape(tf.concat(video, 1), [-1, args.seq_len, args.embed_size]) # 8x20x32
        # lstm_data = tf.concat(axis=2, values=[embeds, visual_feeds])
        # lstm_data = tf.split(lstm_data, args.seq_len, axis=1)
        # lstm_data = [tf.squeeze(d_, [1]) for d_ in lstm_data]
        # h, final_state = seq2seq(video, self.initial_state, cell,
        # )

        # h, final_state = seq2seq(self.video, self.initial_state, cell, scope='rnnlm')
        h, final_state = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=self.video_reshaped,
                                           initial_state=self.initial_state,
                                           dtype=tf.float32,
                                           )
        h = tf.reshape(tf.concat(h, 1), [-1, args.rnn_size]) # 96x32

        # MLP PART GO'S HERE
        outputs = tf.contrib.layers.fully_connected(tf.concat(axis=1, values=[h, embeds_norm]),
                                                     num_outputs=5,
                                                     activation_fn=None)
        # outputs = tf.contrib.layers.fully_connected(outputs, num_outputs=5,
        #                                             activation_fn=tf.nn.tanh)
        #     inputs=tf.concat(axis=1, values=[h, embeds]),
        #     num_outputs=args.output_dim, activation_fn=None)
        # outputs = tf.nn.xw_plus_b(outputs, output_w, output_b)
        flat_target = tf.reshape(self.target_data, [-1, 2])
        [x_data, y_data] = tf.split(flat_target, 2, 1)
        self.final_state = final_state

        def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
            # x - [96,1]
            normx = tf.subtract(x, mux)
            normy = tf.subtract(y, muy)
            # Calculate sx*sy
            sxsy = tf.multiply(sx, sy)
            # Calculate the exponential factor
            z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) -\
                2*tf.div(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
            negRho = 1 - tf.square(rho)
            # Numerator
            result = tf.exp(tf.div(-z, 2*negRho))
            # Normalization constant
            denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
            # Final PDF calculation
            result = tf.div(result, denom)
            return result
    
        def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
            step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))
            # Calculate the PDF of the data w.r.t to the distribution
            result0_1 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_2 = tf_2d_normal(tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_3 = tf_2d_normal(x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
            result0_4 = tf_2d_normal(tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr)
            result0 = tf.div(tf.add(tf.add(tf.add(result0_1, result0_2),
                                           result0_3),
                                    result0_4),
                             tf.constant(4.0, dtype=tf.float32, shape=(1, 1)))
            result0 = tf.multiply(tf.multiply(result0, step), step)

            # result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
            epsilon = 1e-10
            result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability
            # result1 = -tf.log(result1)
            return tf.reduce_sum(result1)
    
        def get_coef(outputs):
            z = outputs
            z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, 1)
            z_sx = tf.exp(z_sx)
            z_sy = tf.exp(z_sy)
            z_corr = tf.tanh(z_corr)
            return [z_mux, z_muy, z_sx, z_sy, z_corr]

        [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(outputs)
        self.mux = o_mux
        self.muy = o_muy
        self.sx = o_sx
        self.sy = o_sy
        self.corr = o_corr
        lossfunc = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
        self.cost = tf.div(lossfunc, (args.batch_size * args.seq_len))
        tvars = tf.trainable_variables()
        l2 = 1e-2 * sum(tf.nn.l2_loss(tvar) for tvar in tvars)
        self.cost += l2
        self.train_op = get_optimizer(self.cost, self.lr)

    def sample(self, sess, traj, future_traj, frames, nodes_in, num=20):
       def sample_gaussian_2d(mux, muy, sx, sy, rho):
           mean = [mux, muy]
           cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
           x = np.random.multivariate_normal(mean, cov, 1)
           return x[0][0], x[0][1]

       # states
       
       
       state = sess.run(self.cell.zero_state(1, tf.float32))
       """
       video = sess.run(self.video_reshaped, feed_dict={
           self.input_files : frames[0,:20].reshape(-1,20),
           })

       
       for iter, pos in enumerate(traj[:-1]):
           data = np.zeros((1, 1, 2), dtype=np.float32)
           node_data = np.zeros((1, 1, 2), dtype=np.float32)
           
           data[0, 0, 0] = pos[0]
           data[0, 0, 1] = pos[1]
           feed_dict = ({self.input_data : data,
                         self.initial_state : state,
                         self.video : video[0][iter],
                         self.target_data : future_traj[iter,:].reshape(1, 1, 2),
                         })
           [state] = sess.run([self.final_state], feed_dict=feed_dict)

       ret = traj # observer first 12 frames, 11?
       last_pos = traj[-1]
       prev_data = np.zeros((1, 1, 2), dtype=np.float32)

       prev_data[0, 0, 0] = last_pos[0]  # x
       prev_data[0, 0, 1] = last_pos[1]  # y

       prev_target_data = np.reshape(future_traj[0], (1, 1, 2))

       for t in range(num):
           feed_dict = ({self.input_data: prev_data,
                         self.initial_state: state,
                         self.target_data: prev_target_data,
                         self.input_files : frames[t].reshape(1,1),
           })
           [o_mux, o_muy, o_sx, o_sy, o_corr, state, cost] = sess.run([self.mux,
                                                                       self.muy,
                                                                       self.sx,
                                                                       self.sy,
                                                                       self.corr,
                                                                       self.final_state,
                                                                       self.cost],
                                                                      feed_dict=feed_dict)
           next_x, next_y = sample_gaussian_2d(o_mux[0][0],
                                               o_muy[0][0],
                                               o_sx[0][0],
                                               o_sy[0][0],
                                               o_corr[0][0])
           ret = np.vstack((ret, [next_x, next_y]))
           prev_data[0, 0, 0] = next_x
           prev_data[0, 0, 1] = next_y
       """
       feed_dict = ({self.input_data: traj.reshape(-1, 15, 2),
                     self.target_data: traj.reshape(-1, 15, 2),
                     self.initial_state: state,
                     self.input_files : frames[0,:20].reshape(-1,20),
           })
       [o_mux, o_muy, o_sx, o_sy, o_corr, state, cost] = sess.run([self.mux,
                                                                   self.muy,
                                                                   self.sx,
                                                                   self.sy,
                                                                   self.corr,
                                                                   self.final_state,
                                                                   self.cost],
                                                                   feed_dict=feed_dict)
       prev_data = np.zeros((1, 1, 2), dtype=np.float32)
       last_pos = traj[-1]
       prev_data[0, 0, 0] = last_pos[0]  # x
       prev_data[0, 0, 1] = last_pos[1]  # y
       ret = traj
       for i in range(num):
           next_x, next_y = sample_gaussian_2d(o_mux[0][0],
                                               o_muy[0][0],
                                               o_sx[0][0],
                                               o_sy[0][0],
                                               o_corr[0][0])
           ret = np.vstack((ret, [next_x, next_y]))
           prev_data[0, 0, 0] = next_x
           prev_data[0, 0, 1] = next_y

           
       print(ret.shape)
       return ret
