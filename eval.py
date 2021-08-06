import sys
from PIL import Image

import numpy as np
import tensorflow as tf
import pathlib

import constants
import data
import discriminator_networks as D
import generator_networks as G


def generate(gen, dataset, replace=False):

    dataset.process_split(shuffle=False)

    for masked, train in ([[True, True], [True, False]]):
        masked_data = dataset.get_data(masked=masked, train=train)
        masked_data_iter = masked_data.as_numpy_iterator()

        for path in masked_data_iter:
            og_path = pathlib.Path(str(path, "utf-8"))
            if not og_path.exists():
                continue

            og_path_parts = list(og_path.parts)

            idx = og_path_parts.index("LFW-masked")
            og_path_parts[idx] = "LFW-unmasked"
            new_path = pathlib.Path("\\".join(og_path_parts))

            if new_path.exists() and not replace:
                continue

            masked_face = dataset.process_path(str(og_path.absolute()))
            unmasked_face = gen(masked_face)
            unmasked_face = tf.reshape(unmasked_face, (constants.IMAGE_SIZE, constants.IMAGE_SIZE, -1))

            converted_image = tf.image.convert_image_dtype(unmasked_face, dtype=tf.uint16, saturate=False)
            encoded_png = tf.io.encode_png(converted_image)
            print(new_path)
            tf.io.write_file(str(new_path.absolute()), encoded_png)


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    np.set_printoptions(threshold=sys.maxsize)

    ### SETUP DATASET ###
    dataset = data.Dataset(batch_size=constants.batch_size, base_path_str="../MaskedFaceGeneration/", more_than=0)

    # Loss without caps: 10e-6
    # Loss with caps: 10e-2
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=10e-6) if constants.caps else tf.keras.optimizers.Adam(
        learning_rate=10e-6)

    disc_loss_fn_1 = D.SiameseCaps.TripletLoss(scale=10e-8)
    disc_loss_fn_2 = D.SiameseCaps.BEGANLoss(scale=10e-3)
    gen_loss_pixel = D.SiameseCaps.PixelLoss(scale=10e2)

    gen = G.ResNetGenerator()
    gen.build(input_shape=(None, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
    gen.compile(loss=[disc_loss_fn_1, disc_loss_fn_2, gen_loss_pixel], optimizer=gen_optimizer)
    gen.summary()

    gen_ckpt = tf.train.Checkpoint(optimizer=gen_optimizer, net=gen)
    gen_manager = tf.train.CheckpointManager(gen_ckpt, './tf_ckpts_gen', max_to_keep=3)
    gen_ckpt.restore(gen_manager.latest_checkpoint)

    generate(gen, dataset, replace=True)
