import os
import sys
import argparse
import pickle as pk
from random import shuffle

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Convolution3D
from tensorflow.keras.layers import MaxPooling3D, Dropout, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm

# For reproductivity
seed = 12306
np.random.seed(seed)

class DeepDrug3DBuilder(object):
    """DeepDrug3D network
    Ouput: a model takes a 5D tensor as input and outpus probabilities of the pocket binds to 
    ATP and Heme.
    """
    @staticmethod
    def build(classes=1):
        model = Sequential()
        # Conv layer 1
        model.add(Convolution3D(
            # input_shape = (33, 33, 33, 1),
            filters=64,
            kernel_size=5,
            padding='valid'    # Padding method
            # data_format='channels_first',
        ))
        model.add(LeakyReLU(alpha = 0.1))
        # Dropout 1
        model.add(Dropout(0.2))
        # Conv layer 2
        model.add(Convolution3D(
            filters=64,
            kernel_size=3,
            padding='valid'      # Padding method
            # data_format='channels_first',
        ))
        model.add(LeakyReLU(alpha = 0.1))
        # Maxpooling 1
        model.add(MaxPooling3D(
            pool_size=(2,2,2),
            strides=None,
            padding='valid'     # Padding method
            # data_format='channels_first'
        ))
        # Dropout 2
        model.add(Dropout(0.4))
        # FC 1
        model.add(Flatten())
        model.add(Dense(128)) # TODO changed to 64 for the CAM
        model.add(LeakyReLU(alpha = 0.1))
        # Dropout 3
        model.add(Dropout(0.4))
        # Fully connected layer 2 to shape (1) for 2 classes
        model.add(Dense(classes))
        if classes == 1:
            model.add(Activation('sigmoid'))
        else:
            model.add(Activation('softmax'))
        return model


def argdet():
    args = myargs()
    return args

def myargs():
    parser = argparse.ArgumentParser()                                              
    parser.add_argument('--bs', required = True, type=int, help = 'batch size')
    parser.add_argument('--lr', required = True, type=float, help = 
                        'initial learning rate')
    parser.add_argument('--epoch', required = True, type=int, help = 
                        'number of epochs for taining')
    parser.add_argument('--output', required = False, help = 
                        'location for the model to be saved')
    args = parser.parse_args()
    return args
    
def train_deepdrug(batch_size, lr, epoch, output):
    # optimize gpu memory usage 
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    mdl = DeepDrug3DBuilder.build()
    adam = Adam(
        lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
        decay=0.0, amsgrad=False)
    
    # We add metrics to get more results you want to see
    mdl.compile(
        optimizer=adam, loss='binary_crossentropy',
        metrics=['binary_accuracy'])
    
    # load the data
    with open("../data/tough_c1/control-pocket.grids", "rb") as f:
        grids = list(pk.load(f).values())
    labels = [np.array([0])] * len(grids)
    with open("../data/tough_c1/nucleotide-pocket.grids", "rb") as f:
        grids += list(pk.load(f).values())
    labels += [np.array([1])] * (len(grids) - len(labels))
    temp = list(zip(grids, labels))
    shuffle(temp)
    grids, labels = zip(*temp)
    grids = np.stack(grids, axis=0)
    grids = np.expand_dims(grids, -1)
    labels = np.stack(labels, axis=0)
    print("grids type:", type(grids))
    print("grid type:", type(grids[0]), "shape:", grids[0].shape)
    print("labels type:", type(labels))
    print("label type:", type(labels[0]), "shape:", labels[0].shape)
    # callback function for model checking
    tfCallBack = callbacks.TensorBoard(log_dir='training_logs')
    mdl.fit(
        grids, labels, epochs = epoch, batch_size = batch_size,
        validation_split=0.3, shuffle = True, callbacks = [tfCallBack])
    # save the model
    if output == None:
        mdl.save('deepdrug3d.h5')
    else:
        mdl.save(output)
    
if __name__ == "__main__":
    args = argdet()
    train_deepdrug(args.bs, args.lr, args.epoch, args.output)
