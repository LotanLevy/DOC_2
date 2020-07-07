import tensorflow as tf
import numpy as np


class TrainTestHelper:
    """
    Manage the train step
    """
    def __init__(self, model, optimizer, D_loss_func, C_loss_func, lambd, training=True):
        self.model = model
        self.optimizer = optimizer
        self.D_loss_func = D_loss_func
        self.C_loss_func = C_loss_func
        self.lambd = lambd

        self.D_loss_logger = tf.keras.metrics.Mean(name='D_loss')
        self.C_loss_logger = tf.keras.metrics.Mean(name='C_loss')

        self.training = training



    def get_step(self):

        @tf.function()
        def train_step(ref_inputs, ref_labels, tar_inputs):
            with tf.GradientTape(persistent=True) as tape:
                #Descriptiveness loss
                prediction = self.model(ref_inputs, training=self.training)
            #     D_loss_value = self.D_loss_func(ref_labels, prediction)
            #     self.D_loss_logger(D_loss_value)
            #
            #     # Compactness loss
            #     prediction = self.model(tar_inputs, training=self.training)
            #     C_loss_value = self.C_loss_func(None, prediction)
            #     self.C_loss_logger(C_loss_value)
            #
            #
            # if self.training:
            #     D_grads = tape.gradient(D_loss_value, self.model.trainable_variables)
            #     C_grads = tape.gradient(C_loss_value, self.model.trainable_variables)
            #     assert (len(D_grads) == len(C_grads))
            #
            #     total_gradient = []
            #     for i in range(len(D_grads)):
            #         total_gradient.append(D_grads[i] * (1 - self.lambd) + C_grads[i] * self.lambd)
            #     self.optimizer.apply_gradients(zip(total_gradient, self.model.trainable_variables))

        return train_step


