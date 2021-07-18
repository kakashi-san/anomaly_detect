import sys
import warnings
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()
from data_handler import Data_Hanlder
from lstm import lrelu, _LSTMCells

class LSTM_VAE(object):
    def __init__(self,dataset_name,columns,z_dim,time_steps,outlier_fraction, n_hidden, batch_size, learning_rate, train_iters):
        self.outlier_fraction = outlier_fraction

        self.data_source = Data_Hanlder(dataset_name,columns,time_steps)
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_iters = train_iters
        
        self.input_dim = len(columns)
        self.z_dim = z_dim
        self.time_steps = time_steps
        self.tup_param = (n_hidden, batch_size, learning_rate, train_iters, z_dim, time_steps)
        
        # if not os.path.exists('./processed'):
        #     os.mkdir('./processed')
        #     with open('./processed/params.pkl', 'wb') as fh:
        #         fh.write(self.tup_param)
    
        self.pointer = 0 
        self.anomaly_score = 0
        self.sess = tf.compat.v1.Session()
        self._build_network()
        self.sess.run(tf.global_variables_initializer())
        
    def _build_network(self):
        with tf.compat.v1.variable_scope('ph'):
            self.X = tf.placeholder(tf.float32,shape=[None,self.time_steps,self.input_dim],name='input_X')
        
        with tf.variable_scope('encoder'):
            with tf.variable_scope('lat_mu'):

                mu_fw_lstm_cells = _LSTMCells([self.z_dim],[lrelu])
                mu_bw_lstm_cells = _LSTMCells([self.z_dim],[lrelu])

                (mu_fw_outputs,mu_fw_outputs),_ = tf.nn.bidirectional_dynamic_rnn(
                                                        mu_fw_lstm_cells,
                                                        mu_bw_lstm_cells, 
                                                        self.X, dtype=tf.float32)
                mu_outputs = tf.add(mu_fw_outputs,mu_fw_outputs)
                
            with tf.variable_scope('lat_sigma'):
                sigma_fw_lstm_cells = _LSTMCells([self.z_dim],[tf.nn.softplus])
                sigma_bw_lstm_cells = _LSTMCells([self.z_dim],[tf.nn.softplus])
                (sigma_fw_outputs,sigma_bw_outputs),_ = tf.nn.bidirectional_dynamic_rnn(
                                                            sigma_fw_lstm_cells,
                                                            sigma_bw_lstm_cells, 
                                                            self.X, dtype=tf.float32)
                sigma_outputs = tf.add(sigma_fw_outputs,sigma_bw_outputs)                 
                sample_Z =  mu_outputs + sigma_outputs * tf.random_normal(
                                                        tf.shape(mu_outputs),
                                                        0,1,dtype=tf.float32)                   
        
        with tf.variable_scope('decoder'):
            recons_lstm_cells = _LSTMCells([self.n_hidden,self.input_dim],[lrelu,lrelu])
            self.recons_X,_ = tf.nn.dynamic_rnn(recons_lstm_cells, sample_Z, dtype=tf.float32)
 
        with tf.variable_scope('loss'):
            reduce_dims = np.arange(1,tf.keras.backend.ndim(self.X))
            recons_loss = tf.losses.mean_squared_error(self.X, self.recons_X)
            kl_loss = - 0.5 * tf.reduce_mean(1 + sigma_outputs - tf.square(mu_outputs) - tf.exp(sigma_outputs))
            self.opt_loss = recons_loss + kl_loss
            self.all_losses = tf.reduce_sum(tf.square(self.X - self.recons_X),reduction_indices=reduce_dims)

        with tf.variable_scope('train'):
            self.uion_train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.opt_loss)
            
            
    def train(self):
        for i in range(self.train_iters):            
            this_X = self.data_source.fetch_data(self.batch_size)
            tf.reset_default_graph()
            self.sess.run([self.uion_train_op],feed_dict={
                    self.X: this_X
                    })
            if i % 200 ==0:
                mse_loss = self.sess.run([self.opt_loss],feed_dict={
                    self.X: self.data_source.train
                    })
                print('round {}: with loss: {}'.format(i,mse_loss))
        self.scores = self._arange_score(self.data_source.train)

        return self.scores


        
    
    def _arange_score(self,input_data):       
        input_all_losses = self.sess.run(self.all_losses,feed_dict={
                self.X: input_data                
                })


        return input_all_losses.tolist()

