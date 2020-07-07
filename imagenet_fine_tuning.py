

import numpy as np
import tensorflow as tf
import os
from dataloader import DataLoader
import utils
from Networks.imagenet_traintest import TrainTestHelper
import argparse
from Networks.losses import compactnes_loss
import random






def train(ref_dataloader,tar_dataloader, trainer, validator, batches, max_iteration, print_freq, network):


    trainstep = trainer.get_step()
    valstep = validator.get_step()
    train_dict = {"iteration":[], "train_D_loss": [], "train_C_loss": [], "val_D_loss": [], "val_C_loss": []}

    for i in range(max_iteration):
        ref_batch_x, ref_batch_y = ref_dataloader.read_batch(batches, "train")
        tar_batch_x, tar_batch_y = tar_dataloader.read_batch(batches, "train")

        trainstep(ref_batch_x, ref_batch_y, tar_batch_x, tar_batch_y)
        if i % print_freq == 0:  # validation loss
            ref_batch_x, ref_batch_y = ref_dataloader.read_batch(batches, "val")
            tar_batch_x, tar_batch_y = tar_dataloader.read_batch(batches, "val")

            valstep(ref_batch_x, ref_batch_y, tar_batch_x, tar_batch_y)

            train_dict["iteration"].append(i)
            train_dict["train_D_loss"].append(float(trainer.D_loss_mean.result()))
            train_dict["train_C_loss"].append(float(trainer.C_loss_mean.result()))
            train_dict["val_D_loss"].append(float(validator.D_loss_mean.result()))
            train_dict["val_C_loss"].append(float(validator.C_loss_mean.result()))

            print("iteration {} - train :D loss {}, C loss {}, val :D loss {}, C loss {}".format(i + 1, train_dict["train_D_loss"][-1], train_dict["train_C_loss"][-1], train_dict["val_D_loss"][-1], train_dict["val_C_loss"][-1]))



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

    parser.add_argument('--ref_train_path', type=str, required=True)
    parser.add_argument('--ref_val_path', type=str, required=True)
    parser.add_argument('--ref_test_path', type=str, required=True)
    parser.add_argument('--tar_train_path', type=str, required=True)
    parser.add_argument('--tar_val_path', type=str, required=True)
    parser.add_argument('--tar_test_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.getcwd(), help='The path to keep the output')
    parser.add_argument('--print_freq', '-pf', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchs_num', '-bs', type=int, default=2, help='number of batches')
    parser.add_argument('--train_iterations', '-iter', type=int, default=800, help='The maximum iterations for learning')
    parser.add_argument('--lambd', type=float, default=0.1, help='lambda constant, the impact of the compactness loss')

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    D_loss = tf.keras.losses.CategoricalCrossentropy()
    C_loss = compactnes_loss

    trainer = TrainTestHelper(network, optimizer, D_loss, C_loss, args.lambd, training=True)
    validator = TrainTestHelper(network, optimizer, D_loss, C_loss, args.lambd, training=False)

    test_images, labels = ref_dataloader.read_batch(200, "test")
    save_predicted_results(test_images, labels, network, ref_dataloader.paths_logger["test"], D_loss, "before_training", args.output_path)


    train(ref_dataloader, tar_dataloader, trainer, validator, args.batchs_num, args.train_iterations, args.print_freq, network)
    save_predicted_results(test_images, labels, network, ref_dataloader.paths_logger["test"], D_loss, "after_training", args.output_path)




if __name__ == "__main__":
    main()
