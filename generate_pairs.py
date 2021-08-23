import sys

import numpy as np
import tensorflow as tf
import collections
import random

import constants
import data


def generate_pairs(dataset):

    dataset.process_split(shuffle=True)

    test_data = dataset.get_data(masked=True, train=False)
    test_data = list(test_data.as_numpy_iterator())

    label_images = collections.defaultdict(list)

    for raw_path in test_data:
        string_path = str(raw_path, "utf8")
        path_list = string_path.split("\\")
        label_images[path_list[8]].append(string_path)

    non_unique = collections.defaultdict(list)
    for key, val in label_images.items():
        if len(val) > 1:
            non_unique[key] = val

    list_keys = list(non_unique.keys())

    printed = 0
    while printed < 6000:

        keys = np.random.choice(list_keys, size=10, replace=False)
        for key in keys:
            ind1, ind2 = np.random.choice(
                non_unique[key], size=2, replace=False)
            path1 = ind1.split("\\")
            path2 = ind2.split("\\")

            label = path1[8]
            img_name1 = path1[-1]
            img_name2 = path2[-1]

            num_ext1 = img_name1.split("_")[-1]
            num_ext2 = img_name2.split("_")[-1]

            num1 = num_ext1.split(".")[0]
            num2 = num_ext2.split(".")[0]

            print(f"\n{label}\t{int(num1)}\t{int(num2)}", end="")

        printed += 10

        keys = np.random.choice(list_keys, size=100, replace=False)
        for key in keys:
            img = np.random.choice(non_unique[key])
            pair_key = random.choice(list_keys)
            pair_img = random.choice(label_images[pair_key])

            path1 = img.split("\\")
            path2 = pair_img.split("\\")

            label1 = path1[8]
            label2 = path2[8]
            img_name1 = path1[-1]
            img_name2 = path2[-1]

            num_ext1 = img_name1.split("_")[-1]
            num_ext2 = img_name2.split("_")[-1]

            num1 = num_ext1.split(".")[0]
            num2 = num_ext2.split(".")[0]

            print(f"\n{label1}\t{int(num1)}\t{label2}\t{int(num2)}", end="")

        printed += 100


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    np.set_printoptions(threshold=sys.maxsize)

    ### SETUP DATASET ###
    dataset = data.Dataset(batch_size=constants.batch_size,
                           base_path_str="../MaskedFaceGeneration/", more_than=1)

    generate_pairs(dataset)
