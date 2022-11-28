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

import math


def select(values, n, s='best', k=100):
    """
    n: the number of selected test cases. 
    s: strategy, ['best', 'random', 'kmst', 'gini']
    k: for KM-ST, the number of ranges. 
    """
    ranks = np.argsort(values) 
    
    if s == 'best':
        h = n//2
        return np.concatenate((ranks[:h],ranks[-h:]))
        
    elif s == 'r':
        return np.array(random.sample(list(ranks),n)) 
    
    elif s == 'kmst':
        fol_max = values.max()
        th = fol_max / k
        section_nums = n // k
        indexes = []
        for i in range(k):
            section_indexes = np.intersect1d(np.where(values<th*(i+1)), np.where(values>=th*i))
            if section_nums < len(section_indexes):
                index = random.sample(list(section_indexes), section_nums)
                indexes.append(index)
            else: 
                indexes.append(section_indexes)
                index = random.sample(list(ranks), section_nums-len(section_indexes))
                indexes.append(index)
        return np.concatenate(np.array(indexes))

    # This is for gini strategy. There is little difference from DeepGini paper. See function ginis() in metrics.py 
    else: 
        return ranks[:n]  



def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

        
def batch_indices(batch_nb, data_length, batch_size):
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end
       
    
def model_retrain_base(dataset, X_original, Y_original, X_additional, Y_additional, input_shape, nb_classes):

    tf.reset_default_graph()

    model = dnn(input_shape, nb_classes)
    
    tf.set_random_seed(1234)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, '../models/' + dataset + '/test.model')
    
    
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    
    predictions = model(x)

    X_train = np.concatenate((X_original, X_additional), axis = 0)
    Y_train = np.concatenate((Y_original, Y_additional), axis = 0)
    
    rng = np.random.RandomState([2019, 7, 15])
    
    loss = model_loss(y, predictions)
    train_step = tf.train.AdamOptimizer(learning_rate=0.005)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        train_step = train_step.minimize(loss)
    
    init_all = False
    save = False
    batch_size = 256
    
    with sess.as_default():
        if hasattr(tf, "global_variables_initializer"):
            if init_all:
                tf.global_variables_initializer().run()
            else:
                initialize_uninitialized_global_variables(sess)
        else:
            warnings.warn("Update your copy of tensorflow; future versions of "
                          "guardai_util may drop support for this version.")
            sess.run(tf.initialize_all_variables())

        eval_params = {'batch_size': 256}
        accuracy = model_eval(sess, x, y, predictions, X_train, Y_train, args=eval_params)
        print('Initial Test accuracy: {0}'.format(accuracy))
        
        a_max = 0
        b_max = 0
        for epoch in range(100):
            nb_batches = int(math.ceil(float(len(X_train)) / batch_size))
            assert nb_batches * batch_size >= len(X_train)

            index_shuf = list(range(len(X_train)))
            rng.shuffle(index_shuf)

            prev = time.time()
            for batch in range(nb_batches):
                start, end = batch_indices(batch, len(X_train), batch_size)
                feed_dict = {x: X_train[index_shuf[start:end]],
                             y: Y_train[index_shuf[start:end]]}
                train_step.run(feed_dict=feed_dict)
            assert end >= len(X_train)  # Check that all examples were used
            

    eval_params = {'batch_size': 256}
    accuracy = model_eval(sess, x, y, predictions, X_original, Y_original, args=eval_params)
    print('Retrained Test accuracy {0}'.format(accuracy))
    return sess, x, preds


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


def Em_fairness_(sess, x, preds, twin_list, initial_label_list):
    preds1 = model_argmax(sess, x, preds, twin_list)
    truths = np.argmax(initial_label_list, axis=1)
    consistent_num = np.sum(preds1 == truths)
    return consistent_num/twin_list.shape[0]



# -----------------------
# Load Model and Dataset
# -----------------------

dataset = 'census'
sens_param = 8  # 'sensitive index, index start from 1, 9 for gender, 8 for race.'
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


# --------------
# QuoTe Process
# --------------

EPOCH = 20    # retraining budget 
REQ = 0.8     # 80% empirical fairness requirement

with np.load("./ds_test.npz") as f:
    x1, x2, y = f['x1'], f['x2'], f['y']

for ep in range(EPOCH):
    
    fairness = Em_fairness_(sess, x, preds, x2, y)
    if  fairness > REQ:
        print("Fairness requirement is already reached: %s", str(fairness))
        break

    os.system("python ./fuzzing.py --dataset %s --sens_param %d --model_path ../models" % (dataset, sens_param))

    global_list = np.load('../results/' + dataset + '/' + str(sens_param) + '/global_samples.npy')
    local_list = np.load('../results/' + dataset + '/' + str(sens_param) + '/local_samples.npy')
    total_list = np.concatenate((global_list, local_list), axis=0)
    total_list, total_list_2 = find_discs(total_list, sens_param)
    Y_total_list = np.array([[0, 1] if label else [1, 0] for label in model_argmax(sess, x, preds, total_list_adf)]) 
    
    X_additional = []
    Y_additional = []
    indexes = list(range(total_list.shape[0]))
    # indexes = select(fols, int(total_list.shape[0]*0.2), s='best')    # if enable selection
    X_retrain_1 = total_list
    X_retrain_2 = total_list_2
    Y_retrain = Y_total_list
    for i in indexes:
        X_additional.append(X_retrain_1[i])
        X_additional.append(X_retrain_2[i])
        Y_additional.append(Y_retrain[i])
        Y_additional.append(Y_retrain[i])
    X_additional = np.array(X_additional)
    Y_additional = np.array(Y_additional)

    sess.close()
    sess, x, preds = model_retrain_base(dataset, X_original, Y_original, X_additional, Y_additional, input_shape, nb_classes)
    save_path = '../models/' + dataset + "/test.model"
    saver = tf.train.Saver()
    saver.save(sess, save_path) # update model

    if ep == EPOCH-1:
        fairness = Em_fairness_(sess, x, preds, x2, y)
        if  fairness > REQ:
            print("Fairness requirement is reached: %s", str(fairness))
        else:
            print("Fairness requirement is not reached: %s", str(fairness))
