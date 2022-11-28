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



def check_for_error_condition_x(conf, sess, x, preds, t, sens):
    t = np.array(t).astype("int")
    label = model_argmax(sess, x, preds, np.array([t]))

    # check for all the possible values of sensitive feature
    for val in range(conf.input_bounds[sens-1][0], conf.input_bounds[sens-1][1]+1):
        if val != int(t[sens-1]):
            tnew = copy.deepcopy(t)
            tnew[sens-1] = val
            label_new = model_argmax(sess, x, preds, np.array([tnew]))
            if label_new != label:
                return tnew
    return None


def find_discs(totals, sens_param):
    oldlist = []
    ttlist = []
    for inp in totals:
        tnew = check_for_error_condition_x(conf, sess, x, preds, inp, sens_param)
        if tnew is None:
            continue
        if tnew.any():
            oldlist.append(inp)
            ttlist.append(tnew)
    return np.array(oldlist), np.array(ttlist)



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

eval_params = {'batch_size': 256}
accuracy = model_eval(sess, x, y, preds, X_original, Y_original, args=eval_params)
print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
# current_estimate = get_estimate(conf, sess, x, preds, sens_param, num_trials, samples)


# load unfair sample points 
global_list_adf = np.load('../results/' + dataset + '/' + str(sens_param) + '/global_samples.npy')
local_list_adf = np.load('../results/' + dataset + '/' + str(sens_param) + '/local_samples.npy')
# global_list_folfuzz = np.load('../results/' + dataset + '/' + str(sens_param) + '/global_samples_folfuzz.npy')

total_list_adf = np.concatenate((global_list_adf, local_list_adf), axis=0)
total_list_adf, total_list_adf_2 = find_discs(total_list_adf, sens_param)
Y_total_list_adf = np.array([[0, 1] if label else [1, 0] for label in model_argmax(sess, x, preds, total_list_adf)]) 

# global_list_adf, global_list_adf_2 = find_discs(global_list_adf, sens_param)
# global_list_folfuzz, global_list_folfuzz_2 = find_discs(global_list_folfuzz, sens_param)
# Y_global_list_adf = np.array([[0, 1] if label else [1, 0] for label in model_argmax(sess, x, preds, global_list_adf)])
# Y_global_list_folfuzz = np.array([[0, 1] if label else [1, 0] for label in model_argmax(sess, x, preds, global_list_folfuzz)])

np.savez('../results/'+dataset+'/'+ str(sensitive_param) + '/adf_ds.npz', x1=total_list_adf, x2=total_list_adf_2, y=Y_total_list_adf)
print("Discriminatory Sample Pairs Saved.")
