
import tensorflow as tf
import numpy as np


def compactnes_loss(target, pred):
    n_dim = np.shape(pred)[0]  # number of features vecs
    k_dim = np.shape(pred)[1]  # feature vec dim
    loss = tf.constant(0.0)
    for i in range(0, n_dim):
        mask = np.ones(n_dim)
        mask[i] = 0
        mask = tf.constant(mask)
        others = tf.boolean_mask(pred, mask)

        mean_vec = tf.math.reduce_sum(others, axis=0) / float(n_dim - 1)
        diff = tf.math.subtract(pred[i], mean_vec) / float(n_dim)
        loss = tf.math.add(tf.math.reduce_sum(tf.math.pow(diff, 2)), loss)

    return loss




def compactnes_loss2(target, pred):
    n_dim = np.shape(pred)[0] # number of features vecs
    k_dim = np.shape(pred)[1] # feature vec dim
    dot_sum = tf.constant(0.0)

    sum_vec = tf.reduce_sum(pred, axis=0)


    for i in range(0, n_dim):
        # mask = np.ones(n_dim)
        # mask[i] = 0
        # mask = tf.constant(mask)
        # others = tf.boolean_mask(pred, mask)
        # m_i = others/ float(n_dim - 1)


        m_i = tf.math.subtract(sum_vec, pred[i])/ float(n_dim - 1)
        x_i = pred[i]

        diff = tf.math.subtract(x_i, m_i)
        dot_sum = tf.math.add(tf.math.reduce_sum(tf.math.pow(diff, 2)), dot_sum)

    return dot_sum /(n_dim * k_dim)



class FeaturesLoss:
    def __init__(self, templates_images, model):
        self.templates_features = self.build_templates(templates_images, model)

    def build_templates(self, templates_images, model):
        templates = []
        for i in range(templates_images.shape[0]):
            image = np.expand_dims(templates_images[i], axis=0)
            templates.append(
                np.squeeze(model(image, training=False), axis=0))
        return np.array(templates)

    def __call__(self, labels, preds):
        preds_num = preds.shape[0]
        losses = np.zeros(preds_num)
        for i in range(preds_num):
            distances = []
            for t in range(self.templates.shape[0]):
                distances.append(np.sqrt(float(np.dot(preds[i] - self.templates_features[t],
                                                      preds[i] - self.templates_features[t]))))  # Eucleaden distance
            losses[i] = min(distances)
        return losses

