from sklearn.externals import joblib
import time
import random
import numpy as np
import tensorflow as tf
import sys, os
sys.path.append("../")
from tensorflow.python.platform import flags

from data.census import census_data
from data.credit import credit_data
from data.bank import bank_data
from model.tutorial_models import dnn
from utils.utils_tf import model_prediction, model_loss, model_argmax
from utils.config import census, credit, bank
from model.network import MLP
from model.layer import *
from utils.utils_tf import model_train, model_eval, model_train_xxx
import copy

os.environ["CUDA_VISIBLE_DEVICES"]="2"

FLAGS = flags.FLAGS


def Em_fairness_(sess, x, preds, twin_list, initial_label_list):
    preds1 = model_argmax(sess, x, preds, twin_list)
    truths = np.argmax(initial_label_list, axis=1)
    consistent_num = np.sum(preds1 == truths)
    return consistent_num/twin_list.shape[0]


dataset = 'bank'
sens_param = 1  # 'sensitive index, index start from 1, 9 for gender, 8 for race.'
additive_percentage = 5.0 # 'percentage of original samples to choose id data.'

num_trials = 100
samples = 100

data = {"census": census_data, "credit": credit_data, "bank": bank_data}
data_config = {"census": census, "credit": credit, "bank": bank}
conf = data_config[dataset]

# data preprocessing
X, Y, input_shape, nb_classes = data[dataset]()
X_original = np.array(X)
Y_original = np.array(Y)

tf.reset_default_graph()

# model structure
model = dnn(input_shape, nb_classes)

# tf operation
tf.set_random_seed(1234)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
saver = tf.train.Saver()
saver.restore(sess, '../models/' + dataset + '/test.model')

x = tf.placeholder(tf.float32, shape=input_shape)
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

preds = model(x)


with np.load("./ds_test.npz") as f:
    x1, x2, y = f['x1'], f['x2'], f['y']

print("Fairness on discriminatory sample pairs:", Em_fairness_(sess, x, preds, x2, y))

