import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import constants
import data
import discriminator_networks as D
import generator_networks as G


def train_gen(gen: G.ClassicGenerator, disc: D.SiameseCaps,
              loss_fn, optimizer,
              epochs, steps, batch_size,
              loss_metric,
              dataset,
              log, log_writer):
    datasets, labels = dataset.process_split()

    logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)

    datasets_processed = []
    for dataset in datasets:
        transformed_set = dataset.map(data.Dataset.process_path)
        transformed_set = transformed_set.batch(batch_size=batch_size)
        transformed_set = transformed_set.cache()
        transformed_set = transformed_set.prefetch(16)

        datasets_processed.append(transformed_set)

    train_set_masked = datasets_processed[0]
    train_set_unmasked = datasets_processed[1]
    train_set_labels = labels[0]

    test_set_masked = datasets_processed[2]
    test_set_unmasked = datasets_processed[3]
    test_set_labels = labels[1]

    for epoch in range(epochs):
        masked_set = iter(train_set_masked)
        unmasked_set = iter(train_set_unmasked)

        step = 0
        for _ in tqdm.tqdm(range(steps)):
            anchor = next(masked_set)
            positive = next(unmasked_set)

            with tf.GradientTape() as tape:
                gen_input = disc.faceNet(anchor, training=False)

                negative = gen(gen_input, training=True)

                disc_input = tf.stack((anchor, negative, positive), axis=1)
                vectors = disc(disc_input, training=False)

                loss_value = loss_fn(y_true=None, y_pred=vectors)

            if file_writer is not None:
                with file_writer.as_default():
                    images = np.reshape(disc_input.numpy(), (-1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
                    tf.summary.image(f"training_data_epoch_{epoch}", images, step=step)

            grads = tape.gradient(loss_value, gen.trainable_variables)
            optimizer.apply_gradients(zip(grads, gen.trainable_variables))

            loss_metric.update_state(loss_value)

            if log:
                with log_writer.as_default():
                    tf.summary.scalar("loss_" + str(epoch), loss_value, step=step)

            step += 1

        print("Loss {0} at epoch {1}".format(loss_metric.result(), epoch))

        if log:
            with log_writer.as_default():
                tf.summary.scalar("epoch_loss_gen", loss_metric.result(), step=epoch)

        # validation_accuracy = 0
        # with tf.device("/cpu:0"):
        #     val_indices = data.batch_hard(val_masked, val_unmasked, chatty=True)
        #     val_hard_triplets = data.make_batches(datasets[2],  # test_masked
        #                                           datasets[3],  # test_unmasked
        #                                           val_indices, num_el=num_el, batch_size=batch_size)
        #
        # for x, _ in tqdm.tqdm(val_hard_triplets):
        #     val_y_ = disc(x, training=False)
        #     anchor, positive, negative = tf.split(val_y_, num_or_size_splits=3, axis=1)
        #
        #     anchor = tf.reshape(anchor, shape=(1, -1))
        #     positive = tf.reshape(positive, shape=(1, -1))
        #     negative = tf.reshape(negative, shape=(1, -1))
        #
        #     positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        #     negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        #
        #     validation_accuracy = validation_accuracy + 1 if positive_dist < negative_dist else validation_accuracy
        #
        # elements_classified = min(len(test_set_masked), steps)
        # correctly_classified = elements_classified - len(val_hard_triplets) + validation_accuracy
        # print("Additional correct classifications:", validation_accuracy)
        # print("Correctly classified:", correctly_classified)
        # print(f"Validation accuracy: {correctly_classified / elements_classified * 100}%")
        # if log:
        #     with log_writer.as_default():
        #         tf.summary.scalar("validation_accuracy", correctly_classified / elements_classified * 100, step=epoch)


def train_disc(disc: D.SiameseCaps, loss_fn, optimizer,
               epochs, steps, batch_size,
               loss_metric,
               dataset: data.Dataset,
               log, log_writer, checkpoint_manager: tf.train.CheckpointManager):
    datasets, labels = dataset.process_split()

    datasets_processed = []
    for dataset in datasets:
        transformed_set = dataset.map(data.Dataset.process_path)
        transformed_set = transformed_set.batch(batch_size=batch_size)
        transformed_set = transformed_set.cache()
        transformed_set = transformed_set.prefetch(16)

        datasets_processed.append(transformed_set)

    train_set_masked = datasets_processed[0]
    train_set_unmasked = datasets_processed[1]
    train_set_labels = labels[0]

    test_set_masked = datasets_processed[2]
    test_set_unmasked = datasets_processed[3]
    test_set_labels = labels[1]

    fcNet = disc.faceNet

    for epoch in range(epochs):
        loss_metric.reset_state()

        embeddings_masked = fcNet.predict(train_set_masked, steps=steps, batch_size=batch_size, verbose=True)
        embeddings_unmasked = fcNet.predict(train_set_unmasked, steps=steps, batch_size=batch_size, verbose=True)

        ### VALIDATION ###
        val_masked = fcNet.predict(test_set_masked, steps=steps, batch_size=batch_size, verbose=True)
        val_unmasked = fcNet.predict(test_set_unmasked, steps=steps, batch_size=batch_size, verbose=True)

        with tf.device("/cpu:0"):
            indices = data.batch_hard(embeddings_masked, embeddings_unmasked, chatty=False)

            num_el = len(indices) if steps is None else steps
            hard_triplets = dataset.make_batches(indices, num_el=num_el, batch_size=batch_size, train=True)

        step = 0
        for x, y in tqdm.tqdm(hard_triplets):
            ind_train_step(disc, x, y, loss_fn, optimizer, loss_metric, epoch, step, log, log_writer)
            step += 1

        print("Loss {0} at epoch {1}".format(loss_metric.result(), epoch))

        if checkpoint_manager:
            save_path = checkpoint_manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        if log:
            with log_writer.as_default():
                tf.summary.scalar("epoch_loss_disc", loss_metric.result(), step=epoch)

        validation_accuracy = 0
        with tf.device("/cpu:0"):
            val_indices = data.batch_hard(val_masked, val_unmasked, chatty=False)
            num_el = len(val_indices) if steps is None else steps

            val_hard_triplets = dataset.make_batches(val_indices, num_el=num_el, batch_size=batch_size, train=False)

        for x, _ in tqdm.tqdm(val_hard_triplets):
            val_y_ = disc(x, training=False)
            anchor, positive, negative = tf.split(val_y_, num_or_size_splits=3, axis=1)

            anchor = tf.reshape(anchor, shape=(1, -1))
            positive = tf.reshape(positive, shape=(1, -1))
            negative = tf.reshape(negative, shape=(1, -1))

            positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

            validation_accuracy = validation_accuracy + 1 if positive_dist < negative_dist else validation_accuracy

        elements_classified = min(len(test_set_masked), steps)
        correctly_classified = elements_classified - len(val_hard_triplets) + validation_accuracy
        print("Additional correct classifications:", validation_accuracy)
        print("Correctly classified:", correctly_classified)
        print(f"Validation accuracy: {correctly_classified / elements_classified * 100}%")
        if log:
            with log_writer.as_default():
                tf.summary.scalar("validation_accuracy", correctly_classified / elements_classified * 100, step=epoch)


def ind_train_step(net, x, y, loss_fn, optimizer, loss_metric, epoch, step, log, log_writer):
    with tf.GradientTape() as tape:
        y_ = net(x, training=True)
        loss_value = loss_fn(y_true=y, y_pred=y_)

    grads = tape.gradient(loss_value, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))

    loss_metric.update_state(loss_value)

    if log:
        with log_writer.as_default():
            tf.summary.scalar("loss_" + str(epoch), loss_value, step=step)


def train(disc_1: D.SiameseCaps, disc_loss_fn_1: tf.keras.losses.Loss, disc_optimizer_1: tf.keras.optimizers.Optimizer,
          disc_2: D.FaceCapsNet, disc_loss_fn_2: tf.keras.losses.Loss, disc_optimizer_2: tf.keras.optimizers.Optimizer,
          gen: G.ClassicGenerator, gen_loss_fn: tf.keras.losses.Loss, gen_optimizer: tf.keras.optimizers.Optimizer,
          epochs, steps, batch_size,
          dataset,
          log, log_writer,
          disc_caps_manager, disc_began_manager, gen_manager):

    dataset.process_split()

    disc_caps_metric = tf.keras.metrics.Mean()
    disc_began_metric = tf.keras.metrics.Mean()
    gen_loss_metric_began = tf.keras.metrics.Mean()
    gen_loss_metric_pixel = tf.keras.metrics.Mean()
    gen_loss_metric_triplet = tf.keras.metrics.Mean()

    logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    # file_writer = None

    fcNet = disc_1.faceNet

    min_val = None

    for epoch in range(epochs):
        disc_caps_metric.reset_state()
        disc_began_metric.reset_state()
        gen_loss_metric_began.reset_state()
        gen_loss_metric_pixel.reset_state()
        gen_loss_metric_triplet.reset_state()

        masked_data = dataset.get_data(masked=True, train=True).take(steps).map(dataset.process_path)
        unmasked_data = dataset.get_data(masked=False, train=True).take(steps).map(dataset.process_path)

        print("\nCalculating embeddings for original images:", flush=True)
        embeddings_masked = fcNet.predict(masked_data, steps=steps, batch_size=1, verbose=True)
        embeddings_unmasked = fcNet.predict(unmasked_data, steps=steps, batch_size=1, verbose=True)

        masked_data = dataset.get_data(masked=True, train=False).take(steps).map(dataset.process_path)
        unmasked_data = dataset.get_data(masked=False, train=False).take(steps).map(dataset.process_path)

        ### VALIDATION ###
        val_masked = fcNet.predict(masked_data, steps=steps, batch_size=batch_size, verbose=True)
        val_unmasked = fcNet.predict(unmasked_data, steps=steps, batch_size=batch_size, verbose=True)

        labels = list(dataset.get_labels(train=True).take(steps).as_numpy_iterator())
        embeddings_masked = tf.reshape(embeddings_masked, shape=(steps, -1))
        embeddings_unmasked = tf.reshape(embeddings_unmasked, shape=(steps, -1))
        negative_indices, positive_indices = data.batch_hard(embeddings_masked, embeddings_unmasked, labels)
        num_el = dataset.extract_sets(negative_indices, positive_indices, steps, train=True)

        for step in tqdm.tqdm(range(num_el)):
            anchor, positive_same, positive_diff, negative = dataset.next_triplet()

            anchor = tf.reshape(anchor, shape=(batch_size, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
            positive_same = tf.reshape(positive_same, shape=(batch_size, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
            positive_diff = tf.reshape(positive_diff, shape=(batch_size, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
            negative = tf.reshape(negative, shape=(batch_size, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

            train_step(disc_1, disc_loss_fn_1, disc_optimizer_1,
                       disc_2, disc_loss_fn_2, disc_optimizer_2,
                       gen, gen_loss_fn, gen_optimizer,
                       anchor, positive_same, positive_diff, negative,
                       disc_caps_metric, disc_began_metric, gen_loss_metric_pixel, gen_loss_metric_triplet,
                       gen_loss_metric_began,
                       file_writer, step, epoch)

        print("Loss classifier caps: {0} at epoch {1}".format(disc_caps_metric.result(), epoch))
        print("Loss classifier began: {0} at epoch {1}".format(disc_began_metric.result(), epoch))
        print("Loss generator: {0} at epoch {1}".format(
            gen_loss_metric_pixel.result() + gen_loss_metric_triplet.result() + gen_loss_metric_began.result(), epoch))

        if log:
            with log_writer.as_default():
                tf.summary.scalar("epoch_loss_disc_caps", disc_caps_metric.result(), step=epoch)
                tf.summary.scalar("epoch_loss_disc_began", disc_began_metric.result(), step=epoch)
                tf.summary.scalar("epoch_loss_gen_began", gen_loss_metric_began.result(), step=epoch)
                tf.summary.scalar("epoch_loss_gen_pixel", gen_loss_metric_pixel.result(), step=epoch)
                tf.summary.scalar("epoch_loss_gen_triplet", gen_loss_metric_triplet.result(), step=epoch)
                tf.summary.scalar("epoch_loss_gen_total",
                                  gen_loss_metric_triplet.result() + gen_loss_metric_pixel.result() + gen_loss_metric_began.result(), step=epoch)

        gen_loss_metric_began.reset_state()
        gen_loss_metric_pixel.reset_state()
        gen_loss_metric_triplet.reset_state()

        ### VALIDATION ###
        labels = list(dataset.get_labels(train=False).take(steps).as_numpy_iterator())
        val_len = len(labels)

        embeddings_masked = tf.reshape(val_masked, shape=(val_len, -1))
        embeddings_unmasked = tf.reshape(val_unmasked, shape=(val_len, -1))
        negative_indices, positive_indices = data.batch_hard(embeddings_masked, embeddings_unmasked, labels)
        num_el = dataset.extract_sets(negative_indices, positive_indices, val_len, train=False)

        for step in tqdm.tqdm(range(num_el)):
            anchor, positive, _, negative = dataset.next_triplet()

            anchor = tf.reshape(anchor, shape=(batch_size, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
            positive = tf.reshape(positive, shape=(batch_size, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

            generated_image = gen.predict(anchor)

            # get the triplet embeddings
            disc_input = tf.stack((anchor, generated_image, positive), axis=1)
            disc_output_id = disc_1(disc_input)

            # get the reconstruction of generated_image
            disc_output_fake = disc_2(generated_image, training=True)

            # get the reconstruction of positive image
            disc_output_real = disc_2(positive, training=True)

            disc_output_real = tf.reshape(disc_output_real, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
            disc_output_fake = tf.reshape(disc_output_fake, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
            positive_same = tf.reshape(positive, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
            y = tf.concat([positive_same, disc_output_real], axis=0)
            x = tf.concat([positive_same, disc_output_fake], axis=0)

            # follow down gradient for generator
            gen_loss_began = tf.keras.backend.relu(disc_loss_fn_2(y_true=y, y_pred=x))
            gen_loss_pixel = tf.reduce_mean(tf.square((tf.subtract(positive, generated_image))))
            gen_loss_triplet = tf.keras.backend.relu(-gen_loss_fn(None, disc_output_id))

            gen_loss_metric_began.update_state(gen_loss_began)
            gen_loss_metric_pixel.update_state(gen_loss_pixel)
            gen_loss_metric_triplet.update_state(gen_loss_triplet)


            if file_writer is not None and step % 100 == 0:
                with file_writer.as_default():
                    images = np.reshape(disc_input.numpy(), (-1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
                    tf.summary.image(f"validation_data_epoch{epoch}", images, step=step)

        total_metric = gen_loss_metric_triplet.result() + gen_loss_metric_pixel.result() + gen_loss_metric_began.result()

        if log:
            with log_writer.as_default():
                tf.summary.scalar("validation_loss_gen_total", gen_loss_metric_began.result() +
                                  gen_loss_metric_pixel.result() +
                                  gen_loss_metric_triplet.result(), step=epoch)
                tf.summary.scalar("validation_loss_gen_began", gen_loss_metric_began.result(), step=epoch)
                tf.summary.scalar("validation_loss_gen_pixel", gen_loss_metric_pixel.result(), step=epoch)
                tf.summary.scalar("validation_loss_gen_triplet", gen_loss_metric_triplet.result(), step=epoch)

        if min_val is None or total_metric < min_val:

            if disc_triplet_manager:
                save_path = disc_triplet_manager.save()
                print("Saved discriminator triplet checkpoint for epoch {}: {}".format(epoch, save_path))

            if disc_began_manager:
                save_path = disc_began_manager.save()
                print("Saved discriminator began checkpoint for epoch {}: {}". format(epoch, save_path))

            if gen_manager:
                save_path = gen_manager.save()
                print("Saved generator checkpoint for epoch {}: {}". format(epoch, save_path))

            min_val = total_metric


def train_step(disc_1: D.SiameseCaps, disc_loss_fn_1: tf.keras.losses.Loss, disc_optimizer_1: tf.keras.optimizers.Optimizer,
               disc_2: D.FaceCapsNet, disc_loss_fn_2: tf.keras.losses.Loss, disc_optimizer_2: tf.keras.optimizers.Optimizer,
               gen: tf.keras.Model, gen_loss_fn: tf.keras.losses.Loss, gen_optimizer: tf.keras.optimizers.Optimizer,
               anchor, positive_same, positive_diff, negative,
               disc_caps_metric: tf.keras.metrics.Metric, disc_began_metric: tf.keras.metrics.Metric,
               gen_loss_metric_pixel: tf.keras.metrics.Metric, gen_loss_metric_triplet: tf.keras.metrics.Metric,
               gen_loss_metric_began: tf.keras.metrics.Metric,
               file_writer, step, epoch):

    # generate a "fake" sample
    generated_image = gen(anchor, training=False)

    ### TRAIN DISCRIMINATOR AGENT 1 ###
    with tf.GradientTape(watch_accessed_variables=False) as disc_tape:
        disc_tape.watch(disc_1.variables)

        # train the discriminator on fake sample
        disc_input = tf.stack((anchor, positive_diff, generated_image), axis=1)
        disc_output_fake = disc_1(disc_input, training=True)

        # # train the discriminator in real samples
        disc_input = tf.stack((anchor, positive_diff, negative), axis=1)
        disc_output_id = disc_1(disc_input, training=True)

        if file_writer is not None and step % 100 == 0:
            with file_writer.as_default():
                images = np.reshape(disc_input.numpy(), (-1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
                tf.summary.image(f"disc_data_epoch_{epoch}", images, step=step)

        # follow down gradient for discriminator
        disc_loss_fake = disc_loss_fn_1(None, disc_output_fake)
        disc_loss_real = disc_loss_fn_1(None, disc_output_id)
        disc_total_loss = tf.keras.backend.relu(disc_loss_real + disc_loss_fake)

    disc_caps_metric.update_state(disc_total_loss)

    disc_grads = disc_tape.gradient(disc_total_loss, disc_1.trainable_variables)
    disc_optimizer_1.apply_gradients(zip(disc_grads, disc_1.trainable_variables))

    ### TRAIN DISCRIMINATOR AGENT 2 ###
    with tf.GradientTape(watch_accessed_variables=False) as disc_tape:
        disc_tape.watch(disc_2.variables)

        # train the discriminator on fake sample
        disc_output_fake = disc_2(generated_image, training=True)

        # # train the discriminator in real samples
        disc_output_real = disc_2(positive_same, training=True)

        if file_writer is not None and step % 100 == 0:
            with file_writer.as_default():
                images = np.reshape(disc_input.numpy(), (-1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
                tf.summary.image(f"disc_data_epoch_{epoch}", images, step=step)

        disc_loss_fn_2.update_lr(disc_2.optimizer.lr)

        disc_output_real = tf.reshape(disc_output_real, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        disc_output_fake = tf.reshape(disc_output_fake, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        positive_same = tf.reshape(positive_same, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        y = tf.concat([positive_same, disc_output_real], axis=0)
        x = tf.concat([positive_same, disc_output_fake], axis=0)

        # follow down gradient for discriminator
        disc_loss = disc_loss_fn_2(y_true=y, y_pred=x)

    disc_began_metric.update_state(disc_loss)

    disc_grads = disc_tape.gradient(disc_loss, disc_2.trainable_variables)
    disc_optimizer_2.apply_gradients(zip(disc_grads, disc_2.trainable_variables))

    ### TRAIN GENERATOR ###
    with tf.GradientTape(watch_accessed_variables=False) as gen_tape:
        gen_tape.watch(gen.variables)

        # generate a "fake" sample
        generated_image = gen(anchor, training=True)

        disc_input = tf.stack((anchor, positive_diff, generated_image), axis=1)

        if file_writer is not None and step % 100 == 0:
            with file_writer.as_default():
                images = np.reshape(disc_input.numpy(), (-1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
                tf.summary.image(f"gen_data_epoch_{epoch}", images, step=step)

        disc_loss_fn_2.update_lr(disc_2.optimizer.lr)

        disc_output_real = tf.reshape(disc_output_real, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        disc_output_fake = tf.reshape(disc_output_fake, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        positive_same = tf.reshape(positive_same, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        y = tf.concat([positive_same, disc_output_real], axis=0)
        x = tf.concat([positive_same, disc_output_fake], axis=0)

        # follow down gradient for generator
        gen_loss_pixel = tf.reduce_mean(tf.square((tf.subtract(positive_same, generated_image))))
        gen_loss_triplet = tf.keras.backend.relu(-gen_loss_fn(None, disc_output_id))
        gen_loss_began = tf.keras.backend.relu(disc_loss_fn_2(y_true=y, y_pred=x))
        gen_loss_total = gen_loss_pixel + gen_loss_began + gen_loss_triplet

    gen_loss_metric_began.update_state(gen_loss_began)
    gen_loss_metric_pixel.update_state(gen_loss_pixel)
    gen_loss_metric_triplet.update_state(gen_loss_triplet)

    gen_grads = gen_tape.gradient(gen_loss_total, gen.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads, gen.trainable_variables))


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)

    np.set_printoptions(threshold=sys.maxsize)

    ### SETUP DATASET ###
    dataset = data.Dataset(batch_size=constants.batch_size, base_path_str="../MaskedFaceGeneration/")

    ### SETUP LOGGING ###
    # logdir = "logs/scalars/" + constants.type_str + "-mid-" + str(constants.mid_caps) + "-" + str(constants.dims) + \
    #          "-out-" + str(constants.output_caps) + "-" + str(constants.dims)
    # histdir = "logs/hist/" + constants.type_str + "-mid-" + str(constants.mid_caps) + "-" + str(constants.dims) + \
    #           "-out-" + str(constants.output_caps) + "-" + str(constants.dims)
    # log_writer = tf.summary.create_file_writer(logdir) if constants.log else None
    # hist_writer = tf.summary.create_file_writer(histdir) if constants.log_vars else None

    # Loss without caps: 10e-6
    # Loss with caps: 10e-2
    disc_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=10e-8) if constants.caps else tf.keras.optimizers.Adam(
        learning_rate=10e-8)
    disc_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=10e-8) if constants.caps else tf.keras.optimizers.Adam(
        learning_rate=10e-8)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=10e-6) if constants.caps else tf.keras.optimizers.Adam(
        learning_rate=10e-6)

    disc_loss_fn_1 = D.SiameseCaps.TripletLoss(scale=10e10)
    disc_loss_fn_2 = D.SiameseCaps.BEGANLoss(scale=10e-3)
    gen_loss_triplet = D.SiameseCaps.TripletLoss(gen=True, scale=10e10)
    gen_loss_pixel = D.SiameseCaps.PixelLoss()

    disc_1 = D.SiameseCaps(mid_size=constants.mid_caps, dims=constants.dims, output_size=constants.output_caps,
                           caps=constants.caps, hist_writer=None)
    disc_1.build(input_shape=(None, 3, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
    disc_1.summary()
    disc_1.compile(loss=disc_loss_fn_1, optimizer=disc_optimizer_1)

    disc_2 = D.FaceCapsNet(mid_size=constants.mid_caps, output_size=constants.output_caps, dims=constants.dims,
                           hist_writer=None, reconstruct=True)
    disc_2.build(input_shape=(None, 1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
    disc_2.summary()
    disc_2.compile(loss=disc_loss_fn_2, optimizer=disc_optimizer_2)

    gen = G.ResNetGenerator()
    gen.build(input_shape=(None, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
    gen.compile(loss=[gen_loss_triplet, disc_loss_fn_2, gen_loss_pixel], optimizer=gen_optimizer)
    gen.summary()

    tf.keras.utils.plot_model(disc_1.model(), "disc_1.png", show_shapes=True)
    tf.keras.utils.plot_model(disc_2.model(), "disc_2.png", show_shapes=True)
    tf.keras.utils.plot_model(gen.model(), "gen.png", show_shapes=True)

    exit(0)
    ### RESTORE ###
    restore_disc_triplet = False
    restore_disc_began = False
    restore_gen = False

    disc_triplet_ckpt = tf.train.Checkpoint(optimizer=disc_optimizer_1, net=disc_1)
    disc_triplet_manager = tf.train.CheckpointManager(disc_triplet_ckpt, './tf_triplet_ckpt', max_to_keep=3)

    if restore_disc_triplet:
        disc_triplet_ckpt.restore(disc_triplet_manager.latest_checkpoint)

    disc_began_ckpt = tf.train.Checkpoint(optimizer=disc_optimizer_2, net=disc_2)
    disc_began_manager = tf.train.CheckpointManager(disc_triplet_ckpt, './tf_began_ckpt', max_to_keep=3)

    if restore_disc_began:
        disc_began_ckpt.restore(disc_began_manager.latest_checkpoint)

    gen_ckpt = tf.train.Checkpoint(optimizer=gen_optimizer, net=gen)
    gen_manager = tf.train.CheckpointManager(gen_ckpt, './tf_ckpts_gen', max_to_keep=3)

    if restore_gen:
        gen_ckpt.restore(gen_manager.latest_checkpoint)

    fcNet = disc_1.faceNet

    # train_disc(disc=disc, loss_fn=disc_loss_fn, optimizer=disc_optimizer,
    #            epochs=15, steps=steps, batch_size=batch_size,
    #            loss_metric=disc_loss_metric, dataset=dataset,
    #            log=log, log_writer=log_writer, checkpoint_manager=manager)

    train(disc_1=disc_1, disc_loss_fn_1=disc_loss_fn_1, disc_optimizer_1=disc_optimizer_1,
          disc_2=disc_2, disc_loss_fn_2=disc_loss_fn_2, disc_optimizer_2=disc_optimizer_2,
          gen=gen, gen_loss_fn=gen_loss_triplet, gen_optimizer=gen_optimizer,
          epochs=constants.epochs, steps=constants.steps, batch_size=constants.batch_size,
          dataset=dataset,
          log=constants.log, log_writer=log_writer,
          disc_caps_manager=disc_triplet_manager, disc_began_manager=disc_began_manager, gen_manager=gen_manager)

    # train_gen(gen=gen, disc=disc, loss_fn=disc_loss_fn, optimizer=gen_optimizer,
    #           epochs=500, steps=50, batch_size=batch_size,
    #           loss_metric=gen_loss_metrics, dataset=dataset,
    #           log=log, log_writer=log_writer)
