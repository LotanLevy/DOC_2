from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.models import Model





import tensorflow as tf


class DOCModel(NNInterface):
    def __init__(self):
        super().__init__()
        self.__model = vgg16.VGG16(weights='imagenet')
        self.__model.summary()


    def call(self, x, training=True):
        x = vgg16.preprocess_input(x)
        return self.__model(x, training=training)

    def compute_output_shape(self, input_shape):
        return self.__model.compute_output_shape(input_shape)

    def get_model_by_output_layer(self, layer_name):
        layer = self.network.get_layer(layer_name).output
        model = Model(self.network.input, outputs=layer)
        return model

    def freeze_layers(self, freeze_idx):

        for i, layer in enumerate(self.__model.layers):
            if freeze_idx > i:
                layer.trainable = False

        for i, layer in enumerate(self.__model.layers):
            print("layer {} is trainable {}".format(layer.name, layer.trainable))