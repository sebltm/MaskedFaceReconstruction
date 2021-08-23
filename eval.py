import collections
import sys

import os
import numpy as np
import tensorflow as tf
import pathlib
import tqdm
import argparse

import constants
import data
import discriminator_networks as D
import eval.facenet.src.facenet
import generator_networks as G
import eval.facenet


def generate(gen, dataset, folder="LFW-unmasked-4", replace=False):
    for path in tqdm.tqdm(dataset):
        if not path.exists():
            continue

        og_path_parts = list(path.parts)

        idx = og_path_parts.index("LFW-masked")
        og_path_parts[idx] = folder
        new_path = pathlib.Path("\\".join(og_path_parts))

        if new_path.exists() and not replace:
            continue

        masked_face = data.Dataset.process_path(str(path.absolute()))
        unmasked_face = gen(masked_face)
        unmasked_face = tf.reshape(
            unmasked_face, (constants.IMAGE_SIZE, constants.IMAGE_SIZE, -1))

        converted_image = tf.image.convert_image_dtype(
            unmasked_face, dtype=tf.uint16, saturate=False)
        encoded_png = tf.io.encode_png(converted_image)
        tf.io.write_file(str(new_path.absolute()), encoded_png)


def read_pairs(file, folder_1):
    paths = set()
    total_paths = []

    with open(file, "r") as f:
        for line in f.readlines()[1:]:
            line = line.split()
            if len(line) == 3:
                label1 = label2 = line[0]
                num1 = int(line[1])
                num2 = int(line[2])
            elif len(line) == 4:
                label1 = line[0]
                label2 = line[2]
                num1 = int(line[1])
                num2 = int(line[3])
            else:
                raise (IncorrectLineSize("Incorrect Size" + str(len(line))))

            base_path = "C:\\Users\\seblt\\OneDrive\\KCL\\Individual_Project\\MaskedFaceGeneration\\"
            fold_path = base_path + folder_1 + "\\"
            img1 = "{0}\\{0}_{1:0>4d}.jpg".format(label1, num1)
            img2 = "{0}\\{0}_{1:0>4d}.jpg".format(label2, num2)

            img1_path = pathlib.Path(fold_path + img1)
            img2_path = pathlib.Path(fold_path + img2)

            assert img1_path.exists(), img1_path
            assert img2_path.exists(), img2_path

            paths.add(img1_path)
            paths.add(img2_path)

            total_paths.append(img1_path)
            total_paths.append(img2_path)

    return paths


class IncorrectLineSize(Exception):
    pass


def load_model(dir):
    dir_exp = os.path.expanduser(dir)
    print("Model directory: {}".format(dir))

    with tf.compat.v1.Session() as sess:
        meta_file, ckpt_file = eval.facenet.src.facenet.get_model_filenames(dir_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        metagraph = tf.compat.v1.train.import_meta_graph(os.path.join(dir_exp, meta_file))

        print(metagraph)

        metagraph.restore(sess, ckpt_file)

    return None


def process_embeddings(model, paths):
    embeddings = collections.defaultdict(tuple)

    for key, val in paths.items():
        data.Dataset.process_path(val, image_size=160)
        embeddings[key] = model(paths[key])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument("--folder_in", type=str, help="Folder to evaluate",
                        default="LFW-masked")
    parser.add_argument("--folder_out", type=str, help="Output folder",
                        default="LFW-unmasked-4")
    return parser.parse_args(argv)


if __name__ == "__main__":

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    sys.path.insert(0, ".\\eval\\facenet")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    np.set_printoptions(threshold=sys.maxsize)

    ### SETUP DATASET ###
    dataset = data.Dataset(batch_size=constants.batch_size,
                           base_path_str="../MaskedFaceGeneration/", more_than=0)

    # Loss without caps: 10e-6
    # Loss with caps: 10e-2
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=10e-6) if constants.caps else tf.keras.optimizers.Adam(
        learning_rate=10e-6)

    disc_loss_fn_1 = D.SiameseCaps.TripletLoss(scale=10e-8)
    disc_loss_fn_2 = D.SiameseCaps.BEGANLoss(scale=10e-3)
    gen_loss_pixel = D.SiameseCaps.PixelLoss(scale=10e2)

    gen = G.ResNetGenerator()
    gen.build(input_shape=(None, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
    gen.compile(loss=[disc_loss_fn_1, disc_loss_fn_2,
                      gen_loss_pixel], optimizer=gen_optimizer)
    gen.summary()

    gen_ckpt = tf.train.Checkpoint(optimizer=gen_optimizer, net=gen)
    gen_manager = tf.train.CheckpointManager(
        gen_ckpt, 'C:\\Users\\seblt\\Local_Projects\\KCL\\pretrained\\tf_ckpts_gen', max_to_keep=3)

    print(gen_manager.latest_checkpoint)
    gen_ckpt.restore(gen_manager.latest_checkpoint)

    args = parse_arguments(sys.argv[1:])
    paths = read_pairs(args.lfw_pairs, folder_1=args.folder_in)

    generate(gen, paths, folder=args.folder_out, replace=True)