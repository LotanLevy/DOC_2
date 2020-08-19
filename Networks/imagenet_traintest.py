import tensorflow as tf
import numpy as np


class TrainTestHelper:
    """
    Manage the train step
    """
    def __init__(self, model, optimizer, D_loss, C_loss, lambd, training=True):
        self.model = model
        self.optimizer = optimizer
        self.D_loss_func = D_loss
        self.C_loss_func = C_loss

        self.D_loss_mean = tf.keras.metrics.Mean(name='D_loss')
        self.C_loss_mean = tf.keras.metrics.Mean(name='C_loss')
        self.accuracy = tf.keras.metrics.Accuracy(name='accuracy')


        self.lambd = lambd

        self.training = training




    def get_step(self):

        @tf.function()
        def train_step(ref_inputs, ref_labels, tar_inputs, tar_labels):
            with tf.GradientTape(persistent=True) as tape:

                # Descriptiveness loss
                prediction = self.model(ref_inputs, training=self.training, ref=True)
                D_loss_value = self.D_loss_func(ref_labels, prediction)
                self.D_loss_mean(D_loss_value)
                self.accuracy.update_state(ref_labels, prediction)

                # Compactness loss
                prediction = self.model(tar_inputs, training=self.training, ref=False)
                C_loss_value = self.C_loss_func(tar_labels, prediction)
                self.C_loss_mean(C_loss_value)

            if self.training:

                D_gradients = tape.gradient(D_loss_value, self.model.trainable_variables)
                C_gradients = tape.gradient(C_loss_value, self.model.trainable_variables)

                total_gradient = []
                assert (len(D_gradients) == len(C_gradients))
                for i in range(len(D_gradients)):
                    total_gradient.append(D_gradients[i] * (1 - self.lambd) + C_gradients[i] * self.lambd)

                self.optimizer.apply_gradients(zip(total_gradient, self.model.trainable_variables))
            return {'D_loss': self.D_loss_mean.result(),
                    'C_loss': self.C_loss_mean.result(),
                    'accuracy': self.accuracy.result()}

        return train_step



