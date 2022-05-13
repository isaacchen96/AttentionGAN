import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


class Resblock(Layer):
    def __init__(self, chn, strides):
        super(Resblock, self).__init__()
        self.chn = chn
        self.strides = strides
        self.conv1 = Conv2D(self.chn, 3, strides=self.strides, padding='same', activation=LeakyReLU(0.3))
        self.bn = BatchNormalization()
        self.conv2 = Conv2D(self.chn, 3, padding='same', activation=LeakyReLU(0.3))
        self.conv3 = Conv2D(self.chn, 3, strides=self.strides, padding='same', activation=None)

    def __call__(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        if inputs.shape[-1] == self.chn:
            return tf.math.add(inputs, conv2)
        else:
            conv3 = self.conv3(inputs)
            return tf.math.add(conv2, conv3)


def build_generator(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(inputs)
    bn = BatchNormalization()(conv1)
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(conv2)
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(conv3)
    for i in range(6):
        resblock = Resblock(256, strides=(1, 1))(bn)
        bn = BatchNormalization()(resblock)

    att_dconv1 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    att_dconv2 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(att_dconv1)
    att1 = Conv2DTranspose(1, 3, strides=(2, 2), padding='same', activation='sigmoid')(att_dconv2)
    att2 = Conv2DTranspose(1, 3, strides=(2, 2), padding='same', activation='sigmoid')(att_dconv2)
    att3 = Conv2DTranspose(1, 3, strides=(2, 2), padding='same', activation='sigmoid')(att_dconv2)

    context_dconv1 = Conv2DTranspose(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    context_dconv2 = Conv2DTranspose(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(context_dconv1)
    context1 = Conv2DTranspose(1, 3, strides=(2, 2), padding='same', activation='tanh')(context_dconv2)
    context2 = Conv2DTranspose(1, 3, strides=(2, 2), padding='same', activation='tanh')(context_dconv2)
    outputs = tf.math.add(att1*context1, att2*context2)
    outputs = tf.math.add(outputs, att3*inputs)
    model = Model(inputs, outputs)
    model.summary()
    return model

def build_discriminator(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(inputs)  # 128 -> 64
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(conv1)  # 64 -> 32
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(conv2)  # 32 -> 16
    conv4 = Conv2D(512, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(conv3)  # 16 -> 8
    flat = Flatten()(conv4)
    classified = Dense(2, activation='softmax')(flat)
    validation = Conv2D(64, 3, strides=(1, 1), padding='same')(conv4)
    validation = Conv2D(2, 3, strides=(1, 1), padding='same')(validation)
    validation = Softmax(axis=-1)(validation)

    model = Model(inputs, [validation, classified])
    model.summary()

    return model


if __name__ == '__main__':
    generator = build_generator()
    discuiminator = build_discriminator()