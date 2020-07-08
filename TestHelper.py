
from plots.plot_helpers import AOC_helper, plot_features
import numpy as np
import os


class TestHelper:
    def __init__(self, ref_dataloader, tar_dataloader, templates_num, test_num, model, output_path):
        self.templates, _, self.templates_paths, self.templates_labels = tar_dataloader.read_batch_with_details(templates_num, "train")
        self.targets, _, self.tar_paths, self.tar_labels = tar_dataloader.read_batch_with_details(test_num, "test")
        self.aliens, _, self.ref_paths, self.ref_labels = ref_dataloader.read_batch_with_details(test_num, "test")
        self.model = model
        self.output_path = output_path

    def get_roc_aoc(self):
        return AOC_helper.get_roc_aoc(self.templates, self.targets, self.aliens, self.model)

    def plot_features(self, full_path, title):
        plot_features(self.templates, self.targets, self.aliens, self.model, full_path, title)

    def find_Optimal_Cutoff(self, tpr, fpr, thresholds):
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

    def calculate_labels_from_scores(self, scores, threshold):
        labels = [int(s < threshold) for s in scores]
        return labels

    def save_prediction(self, fpr, tpr, thresholds, scores, alien=True):
        # fpr, tpr, thresholds, roc_auc, target_dists, alien_dists = AOC_helper.get_roc_aoc_with_scores(self.templates, self.targets, self.aliens, self.model)
        paths = self.ref_paths if alien else self.tar_paths
        # scores = -1 * alien_dists if not alien else -1 * target_dists
        true_labels = np.ones(len(self.ref_paths)) if alien else np.zeros(len(self.tar_paths))
        title = "alien" if alien else "target"

        threshold = self.find_Optimal_Cutoff(tpr, fpr, thresholds)
        predicted_labels = self.calculate_labels_from_scores(scores, threshold)
        with open(os.path.join(self.output_path,"prediction_of_{}.txt".format(title)), 'w') as f:
            f.write("{} {} {}\n".format("path", "true labels", "predicted label"))
            for i in range(len(paths)):
                f.write("{} {} {}\n".format(paths[i], true_labels[i], predicted_labels[i]))

    def save_templates(self):
        with open(os.path.join(self.output_path,"{}.txt".format("templates")), 'w') as f:
            for i in range(len(self.templates_paths)):
                f.write("{} {}\n".format(self.templates_paths[i], self.templates_labels[i]))

    def __del__(self):
        fpr, tpr, thresholds, roc_auc, target_dists, alien_dists = AOC_helper.get_roc_aoc_with_scores(self.templates, self.targets, self.aliens, self.model)

        self.save_prediction(fpr, tpr, thresholds, -1 * alien_dists, alien=True)
        self.save_prediction(fpr, tpr, thresholds, -1*target_dists, alien=False)
        self.save_templates()