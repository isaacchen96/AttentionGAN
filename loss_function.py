import tensorflow as tf
from tensorflow.keras.losses import *

def reconstruction_loss(gt, pred):
    mae = MeanAbsoluteError()
    return mae(gt, pred)

def classify_loss(gt, pred):
    cce = CategoricalCrossentropy()
    return cce(gt, pred)

def adversarial_loss(pred, real=True):
    if real:
        gt = tf.concat([tf.ones((pred.shape[0], pred.shape[1], pred.shape[2], 1), dtype='float32'),
                   tf.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 1), dtype='float32')], axis=-1)
    else:
        gt = tf.concat([tf.zeros((pred.shape[0], pred.shape[1], pred.shape[2], 1), dtype='float32'),
                   tf.ones((pred.shape[0], pred.shape[1], pred.shape[2], 1), dtype='float32')], axis=-1)
    mse = MeanSquaredError()
    return mse(gt, pred)

