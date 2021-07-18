import sys
import warnings
import os

import numpy as np
import tensorflow.compat.v1 as tf


from tensorflow.compat.v1.nn.rnn_cell import MultiRNNCell, LSTMCell
tf.disable_v2_behavior()
def lrelu(x, leak=0.2, name='lrelu'):
	return tf.maximum(x, leak*x)


def _LSTMCells(unit_list,act_fn_list):
    return MultiRNNCell([LSTMCell(unit,                         
                         activation=act_fn) 
                         for unit,act_fn in zip(unit_list,act_fn_list )])