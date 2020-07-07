

import numpy as np
from sklearn.metrics import roc_curve, auc
from Networks.losses import FeaturesLoss
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os






class AOC_helper:
    @staticmethod
    def get_roc_aoc(tamplates, targets, aliens, model):
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

        fpr, tpr, thresholds = roc_curve(labels, scores, 0)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc, np.mean(scores[:target_num]), np.mean(scores[target_num:])


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