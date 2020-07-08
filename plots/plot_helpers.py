

import numpy as np
from sklearn.metrics import roc_curve, auc
from Networks.losses import FeaturesLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from dataloader import read_image, image_name

import os

from PIL import Image

import tensorflow as tf





class AOC_helper:
    @staticmethod
    def get_roc_aoc(tamplates, targets, aliens, model):
        fpr, tpr, thresholds, roc_auc, target_scores, alien_scores = AOC_helper.get_roc_aoc_with_scores(tamplates, targets, aliens, model)

        return fpr, tpr, thresholds, roc_auc, np.mean(target_scores), np.mean(alien_scores)

    @staticmethod
    def get_roc_aoc_with_scores(tamplates, targets, aliens, model):
        loss_func = FeaturesLoss(tamplates, model)

        target_num = len(targets)
        alien_num = len(aliens)

        scores = np.zeros(target_num + alien_num)
        labels = np.zeros(target_num + alien_num)

        preds = model(targets, training=False)
        scores[:target_num] = loss_func(None, preds)
        labels[:target_num] = np.zeros(target_num)

        preds = model(aliens, training=False)
        scores[target_num:] = loss_func(None, preds)
        labels[target_num:] = np.ones(alien_num)

        fpr, tpr, thresholds = roc_curve(labels, -scores, 0)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc, scores[:target_num], scores[target_num:]


class HotMapHelper:
    def __init__(self, tamplates, model, input_size):
        self.loss_func = FeaturesLoss(tamplates, model)
        self.model = model
        self.input_size = input_size

    def test_with_square(self, im_path, kernel_size, stride, output_path):
        im = read_image(im_path, self.input_size)[np.newaxis, :, :, :]
        dim_r, dim_h = int((im.shape[1] - kernel_size) / stride), int((im.shape[2] - kernel_size) / stride)
        scores = np.zeros((dim_r, dim_h))
        i, j = 0, 0
        r, c = int(np.floor(kernel_size / 2)), int(np.floor(kernel_size / 2))
        while r < im.shape[1] - int(np.ceil(kernel_size / 2)):
            while c < im.shape[2] - int(np.ceil(kernel_size / 2)):
                image_cp = im.copy()
                k1, k2 = int(np.floor(kernel_size / 2)), int(np.ceil(kernel_size / 2))
                print(r, c)
                image_cp[0, r - k1: r + k2, c - k1: c + k2, :] = 0

                pred = self.model(image_cp)
                score = -self.loss_func(None, pred)

                scores[i, j] = score
                c += stride
                j += 1
            r += stride
            i += 1
            j = 0
            c = int(np.floor(kernel_size / 2))
        plt.figure()
        ax = sns.heatmap(scores, vmin=np.min(scores), vmax=np.max(scores))
        im_name = image_name(im_path)
        title = "hot_map_of_{}_with_kernel_{}_and_stride_{}".format(im_name, kernel_size, stride)
        plt.title(title)
        plt.savefig(os.path.join(output_path, title + ".png"))
        plt.show()







def plot_features(templates_images, target_images, alien_images, model, full_output_path, title):
    templates_preds = model(templates_images, training=False)
    target_preds = model(target_images, training=False)
    alien_preds = model(alien_images, training=False)

    templates_embedded = TSNE(n_components=2).fit_transform(templates_preds)
    targets_embedded = TSNE(n_components=2).fit_transform(target_preds)
    aliens_embedded = TSNE(n_components=2).fit_transform(alien_preds)
    f = plt.figure()
    plt.scatter(templates_embedded[:, 0], templates_embedded[:, 1], label="templates")
    plt.scatter(targets_embedded[:, 0], targets_embedded[:, 1], label="targets")
    plt.scatter(aliens_embedded[:, 0], aliens_embedded[:, 1], label="aliens")
    plt.legend()

    plt.title(title)
    plt.savefig(full_output_path)
    plt.close(f)


def plot_dict(dict, x_key, output_path):
    for key in dict:
        if key != x_key:
            f = plt.figure()
            plt.plot(dict[x_key], dict[key])
            plt.title(key)
            plt.savefig(os.path.join(output_path, key))
            plt.close(f)
    plt.close("all")



