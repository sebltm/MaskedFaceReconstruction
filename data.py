import pathlib
import sys
import tensorflow as tf
import sklearn.model_selection
from scipy import stats
import tqdm
import numpy as np
import network
import datetime


# def _pairwise_distance(embed1, embed2, square=False):
#     embed1 = tf.reshape(embed1, shape=(tf.shape(embed1)[0], -1))
#     embed2 = tf.reshape(embed2, shape=(tf.shape(embed2)[0], -1))
#
#     dot_product = tf.matmul(embed1, tf.transpose(embed2))
#
#     print(dot_product)
#
#     square_norm = tf.linalg.diag_part(dot_product)
#     distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
#     distances = tf.maximum(distances, 0.0)
#
#     if not square:
#         mask = tf.math.equal(distances, 0.0)
#         mask = tf.dtypes.cast(mask, dtype=tf.float32)
#         distances = distances + mask * 1e-16
#
#         distances = tf.sqrt(distances)
#
#         distances = distances * (1.0 - mask)
#
#     return distances
#
#
# def _get_anchor_negative_triplet_mask(labels):
#     labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
#
#     mask = tf.logical_not(labels_equal)
#
#     return mask
#
#
# def _get_anchor_positive_triplet_mask(labels):
#     indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
#     indices_not_equal = tf.logical_not(indices_equal)
#
#     labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
#
#     mask = tf.logical_and(indices_not_equal, labels_equal)
#
#     return mask
#
#
# def batch_hard(embed1, embed2, labels, square=False):
#     pairwise_dist = _pairwise_distance(embed1, embed2, square=square)
#
#     mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
#     mask_anchor_negative = tf.cast(mask_anchor_negative, tf.float32)
#
#     anchor_negative_dist = tf.multiply(mask_anchor_negative, pairwise_dist)
#
#     max_val_diag = tf.math.reduce_max(anchor_negative_dist, axis=1)
#     anchor_negative_dist = tf.linalg.set_diag(anchor_negative_dist, max_val_diag)
#
#     return tf.math.argmin(anchor_negative_dist, axis=1)

def batch_hard(embed_masked, embed_unmasked, labels, chatty=False):
    embed_masked = tf.reshape(embed_masked, shape=(tf.shape(embed_masked)[0], -1))
    embed_unmasked = tf.reshape(embed_unmasked, shape=(tf.shape(embed_unmasked)[0], -1))

    # norm_masked = tf.norm(embed_masked, axis=1)
    # norm_unmasked = tf.norm(embed_unmasked, axis=1)
    # distance = tf.matmul(embed_masked, embed_unmasked, transpose_b=True)
    # distance = 1 - (distance / (norm_masked * norm_unmasked))

    norm_masked = tf.nn.l2_normalize(embed_masked, axis=1)
    norm_unmasked = tf.nn.l2_normalize(embed_unmasked, axis=-1)
    distance = 1 - tf.matmul(norm_masked, norm_unmasked, transpose_b=True)

    diag = tf.linalg.diag_part(distance)

    dot_product = tf.linalg.set_diag(distance, tf.reduce_min(distance, axis=1))

    mins = tf.reduce_min(dot_product, axis=1)
    indices = tf.math.argmin(dot_product, axis=1)

    mask = mins < diag
    none_array = tf.fill(tf.shape(indices), -1)
    none_array = tf.cast(none_array, tf.int64)

    masked_indices = tf.where(mask, indices, none_array)

    if chatty:
        print(diag.numpy())
        print(mins.numpy())

    return masked_indices


def do_the_thing(i, x):
    print(i, str(x, 'ascii'))
    return x


def make_batches(unmasked_list: tf.data.Dataset, masked_list: tf.data.Dataset, indices, num_el, batch_size=16):
    unmasked_list_enum = list(unmasked_list.take(num_el).as_numpy_iterator())
    masked_list_enum = list(masked_list.take(num_el).as_numpy_iterator())

    unmasked_set = tf.data.Dataset.from_tensor_slices(
        [str(path, 'ascii') for i, path in enumerate(unmasked_list_enum) if indices[i] != -1])
    masked_set = tf.data.Dataset.from_tensor_slices(
        [str(path, 'ascii') for i, path in enumerate(masked_list_enum) if indices[i] != -1])

    negatives_list = np.array(list(unmasked_list.as_numpy_iterator())).take([index for index in indices if index != -1])
    negatives_set = tf.data.Dataset.from_tensor_slices([str(path, 'ascii') for path in negatives_list])

    unmasked_set = unmasked_set.map(Dataset.process_path)
    masked_set = masked_set.map(Dataset.process_path)
    negatives_set = negatives_set.map(Dataset.process_path)

    x_set = tf.data.Dataset.zip((masked_set, unmasked_set, negatives_set)).map(reshape_triplet)

    y_set = tf.zeros(shape=(len(x_set), 3, 10, 8))
    y_set = tf.data.Dataset.from_tensor_slices(y_set)

    complete_set = tf.data.Dataset.zip((x_set, y_set))
    complete_set = complete_set.batch(batch_size=batch_size)

    return complete_set


def reshape_triplet(x1, x2, x3):
    x1 = tf.reshape(x1, shape=(1, 250, 250, 3))
    x2 = tf.reshape(x2, shape=(1, 250, 250, 3))
    x3 = tf.reshape(x3, shape=(1, 250, 250, 3))

    x = tf.concat((x1, x2, x3), axis=0)
    x = tf.expand_dims(x, axis=0)

    return x


def create_lambda():
    yield tf.zeros(shape=(1, 3, 10, 8))


class Dataset:

    def __init__(self, train_test_split=(0.8, 0.2), batch_size=32, base_path_str="../MaskedFaceGeneration"):

        if sum(train_test_split) != 1 or len(train_test_split) != 2:
            raise ValueError("Train test split must be equal to 1")

        self.train_split = train_test_split[0]
        self.batch_size = batch_size

        self.basePath = pathlib.Path(base_path_str)
        self.datasetUnmasked = list(pathlib.Path(f"{str(self.basePath / 'LFW')}").glob("./**/*.jpg"))
        self.datasetMasked = list(pathlib.Path(f"{str(self.basePath / 'LFW-masked')}").glob("./**/*.jpg"))

        self.all_labels = []
        self.le = sklearn.preprocessing.LabelEncoder()
        self._masked_train_data = self._masked_test_data = self._unmasked_train_data = self._unmasked_test_data = []
        self._train_labels = self._test_labels = []
        self.train_split = train_test_split[0]

    def process_split(self):

        train_paths, test_paths = sklearn.model_selection.train_test_split(self.datasetUnmasked,
                                                                           train_size=self.train_split)

        self.all_labels = set()

        masked_train = []
        unmasked_train = []
        train_labels = []

        masked_test = []
        unmasked_test = []
        test_labels = []

        for idx, el in enumerate(zip([train_paths, test_paths],
                                     [[masked_train, unmasked_train, train_labels],
                                      [masked_test, unmasked_test, test_labels]])):

            paths = el[0]

            masked = el[1][0]
            unmasked = el[1][1]
            labels = el[1][2]

            for unmasked_path in paths:

                masked_path = list(unmasked_path.parts)
                index = masked_path.index("LFW")
                masked_path[index] = 'LFW-masked'
                label = masked_path[index + 1]
                masked_path = pathlib.Path("/".join(masked_path))

                if masked_path.exists():
                    masked.append(str(masked_path.resolve()))
                    unmasked.append(str(unmasked_path.resolve()))
                    labels.append(label)
                    self.all_labels.add(label)

        self._masked_train_data = tf.data.Dataset.from_tensor_slices(masked_train)
        self._unmasked_train_data = tf.data.Dataset.from_tensor_slices(unmasked_train)

        self._masked_test_data = tf.data.Dataset.from_tensor_slices(masked_test)
        self._unmasked_test_data = tf.data.Dataset.from_tensor_slices(unmasked_test)

        self._train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
        self._test_labels = tf.data.Dataset.from_tensor_slices(test_labels)

        test_train_sets = [self._masked_train_data, self._unmasked_train_data, self._masked_test_data,
                           self._unmasked_test_data]
        test_train_labels = [self._train_labels, self._test_labels]

        transformed_labels = []
        self.all_labels = list(self.all_labels)

        for label_set in test_train_labels:
            transformed_label = label_set.map(self.decode_label)
            transformed_labels.append(transformed_label)

        return test_train_sets, transformed_labels

    @staticmethod
    def process_path(file_path):
        image_raw = tf.io.read_file(file_path)
        image = tf.image.decode_image(image_raw, dtype=tf.float32)
        img_conv = tf.reshape(image, shape=(250, 250, 3))

        return img_conv

    def decode_label(self, label: tf.Tensor):
        label = tf.argmax(tf.cast(label == self.all_labels, tf.int32))
        return label


if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    np.set_printoptions(threshold=sys.maxsize)

    logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    imgdir = "logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    file_writer = tf.summary.create_file_writer(imgdir)
    log_writer = tf.summary.create_file_writer(logdir)

    batch_size = 10

    dataset = Dataset(batch_size=batch_size, base_path_str="../MaskedFaceGeneration/")
    datasets, labels = dataset.process_split()

    datasets_processed = []
    for dataset in datasets:
        transformed_set = dataset.map(Dataset.process_path)
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

    with file_writer.as_default():
        images = train_set_masked.take(5)
        images = np.reshape(list(images.as_numpy_iterator()), (-1, 250, 250, 3))
        tf.summary.image("Training data", images, step=0)

    optimizer = tf.keras.optimizers.Adam(learning_rate=10e-8)

    net = network.SiameseCaps(output_size=10, caps=True)
    net.build(input_shape=(None, 3, 250, 250, 3))
    net.summary()

    loss_fn = net.TripletLoss()
    net.compile(loss=loss_fn, optimizer='adam')

    fcNet = net.faceNet

    y = tf.data.Dataset.from_generator(create_lambda,
                                       output_signature=(tf.TensorSpec(shape=(1, 10, 8), dtype=tf.float32)))

    steps = int(1500 / batch_size)
    epochs = 1

    tf.profiler.experimental.start('logs/perf')
    for epoch in range(epochs):
        loss_metric = tf.keras.metrics.Mean()
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        embeddings_masked = fcNet.predict(train_set_masked, steps=steps, batch_size=batch_size, verbose=True)
        embeddings_unmasked = fcNet.predict(train_set_unmasked, steps=steps, batch_size=batch_size, verbose=True)

        with tf.device("/cpu:0"):
            indices = batch_hard(embeddings_masked, embeddings_unmasked, list(train_set_labels)[:steps], chatty=False)

            num_el = len(indices) if steps is None else steps
            hard_triplets = make_batches(datasets[0],
                                         datasets[1],
                                         indices, num_el=num_el, batch_size=1)

        step = 0
        for x, y in tqdm.tqdm(hard_triplets):
            with tf.GradientTape() as tape:
                y_ = net(x, training=True)
                loss_value = loss_fn(y_true=y, y_pred=y_)
                grads = tape.gradient(loss_value, net.trainable_variables)

            optimizer.apply_gradients(zip(grads, net.trainable_variables))

            loss_metric.update_state(loss_value)
            # accuracy.update_state(y, net(x, training=True))

            step += 1

        print("Loss {0} at epoch {1}".format(loss_metric.result(), epoch))
        with log_writer.as_default():
            tf.summary.scalar("epoch_loss", loss_value, step=epoch)

        validation_accuracy = 0
        val_masked = fcNet.predict(test_set_masked, steps=steps, batch_size=batch_size, verbose=True)
        val_unmasked = fcNet.predict(test_set_unmasked, steps=steps, batch_size=batch_size, verbose=True)

        with tf.device("/cpu:0"):
            val_indices = batch_hard(val_masked, val_unmasked, list(test_set_labels), chatty=True)
            val_hard_triplets = make_batches(datasets[2],
                                             datasets[3],
                                             val_indices, num_el=num_el)

        for x, _ in tqdm.tqdm(val_hard_triplets):
            val_y_ = net(x, training=False)
            anchor, positive, negative = tf.split(val_y_, num_or_size_splits=3, axis=1)
            anchor = tf.reshape(anchor, shape=(1, -1))
            positive = tf.reshape(positive, shape=(1, -1))
            negative = tf.reshape(negative, shape=(1, -1))

            positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

            validation_accuracy = validation_accuracy + 1 if positive_dist < negative_dist else validation_accuracy

        with log_writer.as_default():
            tf.summary.scalar("validation_accuracy", validation_accuracy / len(indices), step=epoch)
        # print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
        #                                                             loss_metric.result(),
        #                                                             accuracy.result()))
    tf.profiler.experimental.stop()
