

import numpy as np
import tensorflow as tf
import os
from dataloader import DataLoader
import utils
from Networks.imagenet_traintest import TrainTestHelper
import argparse
from Networks.losses import compactnes_loss
import random
from plots.plot_helpers import AOC_helper, plot_features, plot_dict

from dataloader import create_generators




from TestHelper import TestHelper




def train(ref_dataloader,tar_dataloader, trainer, validator, batches, max_iteration, print_freq, test_helper, output_path, network):


    ref_train_gen, ref_val_gen = ref_dataloader

    tar_train_gen, tar_val_gen = tar_dataloader


    trainstep = trainer.get_step()
    valstep = validator.get_step()
    train_dict = {"iteration":[], "train_D_loss": [], "train_C_loss": [], "train_accuracy": [],
                  "val_D_loss": [], "val_C_loss": [], "val_accuracy": []}
    test_dict = {"iteration":[], "auc": [], "target_dists": [], "alien_dists":[]}


    for i in range(max_iteration):
        ref_inputs, ref_labels = ref_train_gen.next()
        tar_inputs, tar_labels = tar_train_gen.next()
        # ref_batch_x, ref_batch_y = ref_dataloader.read_batch(batches, "train")
        # tar_batch_x, tar_batch_y = tar_dataloader.read_batch(batches, "train")
        # val_output = trainstep(ref_batch_x, ref_batch_y, tar_batch_x, tar_batch_y)

        train_output = trainstep(ref_inputs, ref_labels, tar_inputs, tar_labels)
        if i % print_freq == 0:  # validation loss
            ref_inputs, ref_labels = ref_val_gen.next()
            tar_inputs, tar_labels = tar_val_gen.next()

            # ref_batch_x, ref_batch_y = ref_dataloader.read_batch(batches, "val")
            # tar_batch_x, tar_batch_y = tar_dataloader.read_batch(batches, "val")

            val_output = valstep(ref_inputs, ref_labels, tar_inputs, tar_labels)



            train_dict["iteration"].append(i)
            train_dict["train_D_loss"].append(float(train_output['D_loss']))
            train_dict["train_C_loss"].append(float(train_output['C_loss']))
            train_dict["train_accuracy"].append(float(train_output['accuracy']))

            train_dict["val_D_loss"].append(float(val_output['D_loss']))
            train_dict["val_C_loss"].append(float(val_output['C_loss']))
            train_dict["val_accuracy"].append(float(val_output['accuracy']))

            print_output = ""
            for key in train_dict.keys():
                print_output += "{}: {},".format(key, train_dict[key][-1])

            #
            #
            # print("iteration {} - train :"
            #       "D loss {}, C loss {}, val :D loss {}, C loss {}".format(i + 1, train_dict["train_D_loss"][-1], train_dict["train_C_loss"][-1], train_dict["val_D_loss"][-1], train_dict["val_C_loss"][-1]))

        # if i % (2 * print_freq) == 0: # test
        #     test_dict["iteration"].append(i)
        #     test_results = test_helper.get_roc_aoc()
        #     test_dict["auc"].append(test_results[3])
        #     test_dict["target_dists"].append(test_results[4])
        #     test_dict["alien_dists"].append(test_results[5])
        #     test_helper.plot_features(os.path.join(output_path, "features_after_{}_iterations.png".format(i)), "features_after_{}_iterations".format(i))
        #
        #     network.save_model(i, output_path)

    plot_dict(test_dict, "iteration", output_path)
    plot_dict(train_dict, "iteration", output_path)




def get_imagenet_prediction(image, hot_vec,  network, loss_func):
    pred = network(image, training=False)
    i = tf.math.argmax(pred[0])
    loss = loss_func(hot_vec, pred)
    return i, np.array(pred[0])[i], loss

def save_predicted_results(test_images, labels, network, paths, loss_func, title, output_path):
    with open(os.path.join(output_path, "{}.txt".format(title)), 'w') as f:
        correct_sum = 0
        for i in range(len(test_images)):
            pred, score, loss = get_imagenet_prediction(test_images[i][np.newaxis, :,:,:], labels[i], network, loss_func)
            f.write("{} {} {} {}\n".format(paths[i], pred, score, loss))
            if int(pred) == int(np.argmax(labels[i])):
                correct_sum += 1
        f.write("correctness {}\n".format(correct_sum/len(test_images)))



def get_args():
    parser = argparse.ArgumentParser(description='Process training arguments.')
    parser.add_argument('--nntype', default="PerceptualModel", help='The type of the network')
    parser.add_argument('--cls_num', type=int, default=1000, help='The number of classes in the dataset')
    parser.add_argument('--input_size', type=int, nargs=2, default=(224, 224))


    # parser.add_argument('--ckpt', type=str, default=None)


    #
    # parser.add_argument('--ref_train_path', type=str, required=True)
    # parser.add_argument('--ref_val_path', type=str, required=True)
    # parser.add_argument('--ref_test_path', type=str, required=True)
    # parser.add_argument('--tar_train_path', type=str, required=True)
    # parser.add_argument('--tar_val_path', type=str, required=True)
    # parser.add_argument('--tar_test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--print_freq', '-pf', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchs_num', '-bs', type=int, default=2, help='number of batches')
    parser.add_argument('--train_iterations', '-iter', type=int, default=800, help='The maximum iterations for learning')
    parser.add_argument('--lambd', type=float, default=0.1, help='lambda constant, the impact of the compactness loss')
    parser.add_argument('--templates_num', '-tn', type=int, default=40, help='The number pf templates in the testing')
    parser.add_argument('--test_num', type=int, default=100, help='The number of test examples to consider')
    parser.add_argument('--test_layer', '-tl', default="fc2", help='The name of the network layer for the test output')

    parser.add_argument("--ref_path", required=True,
                        help="The directory of the reference dataset")
    parser.add_argument("--tar_path", required=True,
                        help="The directory of the target dataset")

    parser.add_argument("--ref_aug", action='store_true')
    parser.add_argument("--tar_aug", action='store_true')




    return parser.parse_args()

def main():
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)

    tf.keras.backend.set_floatx('float32')
    args = get_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    ref_train_datagen, ref_val_datagen, tar_train_datagen, tar_val_datagen = create_generators(
        args.ref_path, args.tar_path,
        args.ref_aug, args.tar_aug,
        args.input_size, args.batchs_num)

    # ref_dataloader = DataLoader(args.ref_train_path, args.ref_val_path, args.ref_test_path, args.cls_num, args.input_size,
    #                         name="ref_dataloader", output_path=args.output_path)
    # tar_dataloader = DataLoader(args.tar_train_path, args.tar_val_path, args.tar_test_path, args.cls_num, args.input_size,
    #                         name="tar_dataloader", output_path=args.output_path)
    network = utils.get_network(args.nntype)
    network.freeze_layers(19)
    #if args.ckpt is not None:
     #   network.load_weights(args.ckpt).expect_partial()  # expect_partial enables to ignore training information for prediction
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    D_loss = tf.keras.losses.CategoricalCrossentropy()
    C_loss = compactnes_loss
    features_model = network.get_features_model(args.test_layer)



    trainer = TrainTestHelper(network, optimizer, D_loss, C_loss, args.lambd, training=True)
    validator = TrainTestHelper(network, optimizer, D_loss, C_loss, args.lambd, training=False)

    # test_images, labels = ref_dataloader.read_batch(200, "test")
    # save_predicted_results(test_images, labels, network, ref_dataloader.paths_logger["test"], D_loss, "before_training", args.output_path)
    #
    # random.seed(1234)
    # np.random.seed(1234)
    # tf.random.set_seed(1234)
    #
    # test_helper = TestHelper(ref_dataloader, tar_dataloader, args.templates_num, args.test_num, features_model, args.output_path)
    # random.seed(1234)
    # np.random.seed(1234)
    # tf.random.set_seed(1234)


    train(ref_dataloader, tar_dataloader, trainer, validator, args.batchs_num, args.train_iterations, args.print_freq, test_helper, args.output_path, network)
    # save_predicted_results(test_images, labels, network, ref_dataloader.paths_logger["test"], D_loss, "after_training", args.output_path)

    network.save_model(args.train_iterations, args.output_path)




if __name__ == "__main__":
    main()
