from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from attack import FGSM, PGD
os.environ["CUDA_VISIBLE_DEVICES"]="0" 


# Suppress the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
    

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

    else: 
        return ranks[:n]  
    
    
    
def load_mnist(path="./mnist.npz"):
    """
    preprocessing for MNIST dataset, values are normalized to [0,1].  
    y_train and y_test are one-hot vectors. 
    """
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return x_train, x_test, y_train, y_test



# -----------------------
# Load Model and Dataset
# -----------------------

path = "./mnist.npz"
x_train, x_test, y_train, y_test = load_mnist(path)

model_path = "./Lenet5_mnist.h5"
model = keras.models.load_model(model_path)

with np.load("./FGSM_Test.npz") as f:
    fgsm_test, fgsm_test_labels = f['advs'], f['labels']

with np.load("./PGD_Test.npz") as f:
    pgd_test, pgd_test_labels = f['advs'], f['labels']


advx_test = np.concatenate((fgsm_test, pgd_test))
advy_test = np.concatenate((fgsm_test_labels, pgd_test_labels))



# --------------
# RobOT Process
# --------------

EPOCH = 20    # retraining budget 
REQ = 0.8     # 80% empirical robustness requirement

for ep in range(EPOCH):

    _, robustness = model.evaluate(advx_test, advy_test, verbose=0)
    if  robustness > REQ:
        print("Robustness requirement is already reached: %s", str(robustness))
        break

    # os.system("python ./gen_adv.py")
    fgsm = FGSM(model, ep=0.3, isRand=True)
    pgd = PGD(model, ep=0.3, epochs=10, isRand=True)

    # seed_indexes = np.array(random.sample(list(range(x_train.shape[0])),n)) 
    fgsm_train, fgsm_train_labels, fgsm_train_fols, fgsm_train_ginis = fgsm.generate(x_train, y_train)
    pgd_train, pgd_train_labels, pgd_train_fols, pgd_train_ginis = pgd.generate(x_train, y_train)
    fp_train = np.concatenate((fgsm_train, pgd_train))
    fp_train_labels = np.concatenate((fgsm_train_labels, pgd_train_labels))
    
#    fp_train_fols = np.concatenate((fgsm_train_fols, pgd_train_fols))
#    indexes = select(fp_train_fols, int(fp_train.shape[0]*0.2), s='best')   # if enable selection
    indexes = list(range(fp_train.shape[0]))
    selectAdvs = fp_train[indexes]
    selectAdvsLabels = fp_train_labels[indexes]
    x_train_mix = np.concatenate((x_train, selectAdvs),axis=0)
    y_train_mix = np.concatenate((y_train, selectAdvsLabels),axis=0)

    model.fit(x_train_mix, y_train_mix, epochs=5, batch_size=64, verbose=0)
    model.save(model_path)

    if ep == EPOCH-1:
        _, robustness = model.evaluate(advx_test, advy_test, verbose=0)
        if  robustness > REQ:
            print("Robustness requirement is reached: %s", str(robustness))
        else:
            print("Robustness requirement is not reached: %s", str(robustness))


