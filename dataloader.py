

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import random
import re


SPLIT_FACTOR = "$"

def image_name(image_path):
    regex = ".*[\\/|\\\](.*)[\\/|\\\](.*).jpg"
    m = re.match(regex, image_path)
    return m.group(1) + "_" + m.group(2)


def read_image(path, resize_image=()):
    image = Image.open(path, 'r')
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if len(resize_image) > 0:
        image = image.resize(resize_image, Image.NEAREST)
    image = np.array(image).astype(np.float32)
    return image

def read_dataset_map(data_map_path, shuffle=False):
    with open(data_map_path, "r") as lf:
        lines_list = lf.read().splitlines()
        if shuffle:
            random.shuffle(lines_list)
        lines = [line.split(SPLIT_FACTOR) for line in lines_list]
        images, labels = [], []
        if len(lines) > 0:
            images, labels = zip(*lines)
        labels = [int(label) for label in labels]
    return images, np.array(labels).astype(np.int)



class DataLoader:

    def __init__(self, train_file, val_file, test_file, cls_num, input_size, name="dataloader",
                 output_path=os.getcwd()):
        self.classes_num = cls_num
        self.input_size = input_size

        self.name = name
        self.output_path = output_path
        self.paths_logger = {"train": [], "val": [], "test": []}
        self.labels_logger = {"train": [], "val": [], "test": []}

        self.datasets = {"train": read_dataset_map(train_file, shuffle=True),
                       "val": read_dataset_map(val_file, shuffle=True),
                       "test": read_dataset_map(test_file, shuffle=True)}

        self.batches_idx = {"train": 0, "val": 0, "test": 0}

    def read_batch_with_details(self, batch_size, mode):
        all_paths, all_labels = self.datasets[mode]

        indices = list(range(self.batches_idx[mode], min(self.batches_idx[mode] + batch_size, len(all_paths))))
        if len(indices) < batch_size:
            self.batches_idx[mode] = 0
            rest = batch_size - len(indices)
            indices += list(range(self.batches_idx[mode], min(self.batches_idx[mode] + rest, len(all_paths))))

        self.batches_idx[mode] += batch_size

        # rand_idx = np.random.randint(low=0, high=len(all_paths)-1, size=batch_size).astype(np.int)

        batch_labels = all_labels[indices]
        batch_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 3))
        paths = []
        labels = []
        b_idx = 0
        for i in indices:
            batch_images[b_idx, :, :, :] = read_image(all_paths[i], self.input_size)
            paths.append(all_paths[i])
            labels.append(all_labels[i])
            b_idx += 1

        hot_vecs = tf.keras.utils.to_categorical(batch_labels, num_classes=self.classes_num)
        return batch_images, hot_vecs, paths, labels



    def read_batch(self, batch_size, mode):
        batch_images, hot_vecs, paths, labels = self.read_batch_with_details(batch_size, mode)
        self.paths_logger[mode] += paths
        self.labels_logger[mode] += labels
        return batch_images, hot_vecs
        # all_paths, all_labels = self.datasets[mode]
        #
        # indices = list(range(self.batches_idx[mode], min(self.batches_idx[mode] + batch_size, len(all_paths))))
        # if len(indices) < batch_size:
        #     self.batches_idx[mode] = 0
        #     rest = batch_size-len(indices)
        #     indices += list(range(self.batches_idx[mode], min(self.batches_idx[mode] + rest, len(all_paths))))
        #
        # self.batches_idx[mode] += batch_size
        #
        #
        #
        # # rand_idx = np.random.randint(low=0, high=len(all_paths)-1, size=batch_size).astype(np.int)
        #
        # batch_labels = all_labels[indices]
        # batch_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 3))
        # b_idx = 0
        # for i in indices:
        #     batch_images[b_idx, :, :, :] = read_image(all_paths[i], self.input_size)
        #     self.paths_logger[mode].append(all_paths[i])
        #     self.labels_logger[mode].append(all_labels[i])
        #     b_idx += 1
        #
        # hot_vecs = tf.keras.utils.to_categorical(batch_labels, num_classes=self.classes_num)
        # return batch_images, hot_vecs

    def __del__(self):
        for mode in self.paths_logger:
            with open(os.path.join(self.output_path, "{}_{}.txt".format(self.name, mode)), 'w') as f:
                for i in range(len(self.paths_logger[mode])):
                    f.write("{}{}{}\n".format(self.paths_logger[mode][i], SPLIT_FACTOR, self.labels_logger[mode][i]))



def create_generators(ref_path, tar_path, ref_aug, tar_aug, input_size, batch_size):
    ref_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.2)
    ref_classes = [str(i) for i in range(1000)]
    ref_train_datagen = ref_gen.flow_from_directory(ref_path, subset="training",
                                                      seed=123,
                                                      shuffle=True,
                                                      class_mode="categorical",
                                                      target_size=input_size,
                                                      batch_size=batch_size, classes=ref_classes)
    ref_val_datagen = ref_gen.flow_from_directory(ref_path, subset="validation",
                                                      seed=123,
                                                      shuffle=True,
                                                      class_mode="categorical",
                                                      target_size=input_size,
                                                      batch_size=batch_size, classes=ref_classes)


    tar_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=0.2)
    tar_train_datagen = tar_gen.flow_from_directory(tar_path, subset="training",
                                                      seed=123,
                                                      shuffle=True,
                                                      class_mode="categorical",
                                                      target_size=input_size,
                                                      batch_size=batch_size, classes=ref_classes)
    tar_val_datagen = tar_gen.flow_from_directory(tar_path, subset="validation",
                                                      seed=123,
                                                      shuffle=True,
                                                      class_mode="categorical",
                                                      target_size=input_size,
                                                      batch_size=batch_size, classes=ref_classes)

    return ref_train_datagen, ref_val_datagen, tar_train_datagen, tar_val_datagen








