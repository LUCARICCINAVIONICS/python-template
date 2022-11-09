import tensorflow as tf
import keras as k

from keras.layers import Input, Conv2D, Activation, BatchNormalization,Flatten,MaxPooling2D ,Reshape,Conv2DTranspose,LeakyReLU
from keras.layers.core import Dropout,Dense
from keras.models import Sequential
from matplotlib import pyplot as plt




#
# input = input[..., None , None] # keras needs 4D input, so add 1 dimension
class Network():
    def __init__(self):
        mBufferSize = 60000
        mBatchSize = 256
        mImgHeight = 180
        mIngWidth = 180

    def InitParam(self,aBufferSize,aBatchSize,aImgHeight,aImgWidth):
        self.mBufferSize = aBufferSize
        self.mBatchSize = aBatchSize
        self.mImgHeight = aImgHeight
        self.mIngWidth = aImgWidth
      
    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model

    def load(image_file):
        # Read and decode an image file to a uint8 tensor
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image)

        # Split each image tensor into two tensors:
        # - one with a real building facade image
        # - one with an architecture label image 
        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, w:, :]
        real_image = image[:, :w, :]

        # Convert both images to float32 tensors
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image