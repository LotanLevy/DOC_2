from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dropout




import tensorflow as tf


class PerceptualModel(NNInterface):
    def __init__(self):
        super().__init__()
        self.__model = vgg16.VGG16(weights='imagenet')
        self.__model.summary()


    def call(self, x, training=True):
        x = vgg16.preprocess_input(x)
        return self.__model(x, training=training)

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)

    def freeze_layers(self, freeze_idx):

        for i, layer in enumerate(self.__model.layers):
            if freeze_idx > i:
                layer.trainable = False

        for i, layer in enumerate(self.__model.layers):
            print("layer {} is trainable {}".format(layer.name, layer.trainable))

    def add_dropout(self):
        # Store the fully connected layers
        fc1 = self.__model.layers[-3]
        fc2 = self.__model.layers[-2]
        predictions = self.__model.layers[-1]

        # Create the dropout layers
        dropout1 = Dropout(0.5)
        dropout2 = Dropout(0.5)

        # Reconnect the layers
        x = dropout1(fc1.output)
        x = fc2(x)
        # x = dropout2(x)
        predictors = predictions(x)
        input = self.__model.input

        # Create a new model
        self.__model = Model(input, predictors)
        # self.__model.summary()