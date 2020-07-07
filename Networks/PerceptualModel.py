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
        self.ref_model = self.get_dropout_model(2)
        self.tar_model = self.get_dropout_model(1)
        print(self.tar_model.summary())

    def get_features_model(self, layer_name):
        layer = self.__model.get_layer(layer_name).output
        model = Model(self.__model.input, outputs=layer)
        return model


    def call(self, x, training=True, ref=True):
        x = vgg16.preprocess_input(x)
        if ref:
            return self.ref_model(x, training=training)
        else:
            return self.tar_model(x, training=training)

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


    def get_dropout_model(self, dropout_num):
        model = tf.keras.Sequential()

        dropout1 = Dropout(0.5)
        dropout2 = Dropout(0.5)

        for layer in self.__model.layers:
            model.add(layer)
            if layer.name == "fc1":
                model.add(dropout1)
            if layer.name == "fc2" and dropout_num > 1:
                model.add(dropout2)
        return model