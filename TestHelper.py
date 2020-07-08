
from plots.plot_helpers import AOC_helper, plot_features, HotMapHelper
import numpy as np
import os
import argparse
import tensorflow as tf
from dataloader import DataLoader
import random
import utils


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

    def predict_hot_maps(self, im_paths, kernel_size, stride, input_size):
        helper = HotMapHelper(self.templates, self.model, input_size)
        for path in im_paths:
            helper.test_with_square(path, kernel_size, stride, self.output_path)



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





def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="PerceptualModel", help='The type of the network')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--cls_num', type=int, default=1000, help='The number of classes in the dataset')
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))


    parser.add_argument('--ref_train_path', type=str, required=True)
    parser.add_argument('--ref_val_path', type=str, required=True)
    parser.add_argument('--ref_test_path', type=str, required=True)
    parser.add_argument('--tar_train_path', type=str, required=True)
    parser.add_argument('--tar_val_path', type=str, required=True)
    parser.add_argument('--tar_test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--templates_num', '-tn', type=int, default=40, help='The number pf templates in the testing')
    parser.add_argument('--test_num', type=int, default=100, help='The number of test examples to consider')
    parser.add_argument('--test_layer', '-tl', default="fc2", help='The name of the network layer for the test output')


    parser.add_argument('--hot_map_paths',  type=str, default="None")
    parser.add_argument('--kernel_size', '-ks', type=int, default=24)
    parser.add_argument('--stride', '-s', type=int, default=10)

    return parser.parse_args()

def main():
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    tf.keras.backend.set_floatx('float32')
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ref_dataloader = DataLoader(args.ref_train_path, args.ref_val_path, args.ref_test_path, args.cls_num, args.input_size,
                            name="ref_dataloader", output_path=args.output_path)
    tar_dataloader = DataLoader(args.tar_train_path, args.tar_val_path, args.tar_test_path, args.cls_num, args.input_size,
                            name="tar_dataloader", output_path=args.output_path)

    network = utils.get_network(args.nntype)
    network.freeze_layers(19)
    network.load_model(args.ckpt_dir)
    features_model = network.get_features_model(args.test_layer)


    test_helper = TestHelper(ref_dataloader, tar_dataloader, args.templates_num, args.test_num, features_model, args.output_path)

    if args.hot_map_paths != None:
        paths = []
        with open(args.hot_map_paths, "r") as f:
            for line in f:
                paths.append(line.rstrip('\n'))
        test_helper.predict_hot_maps(paths, args.kernel_size, args.stride, args.input_size)




if __name__ == "__main__":
    main()
