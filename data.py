import itertools
import pathlib

import numpy as np
import sklearn.model_selection
import tensorflow as tf

import constants


def batch_hard(embed_masked, embed_unmasked, labels, chatty=False):

    # Get the dot product
    na = tf.matmul(embed_masked, tf.transpose(embed_masked))
    nb = tf.matmul(embed_unmasked, tf.transpose(embed_unmasked))

    square_norm_na = tf.linalg.diag_part(na)
    square_norm_nb = tf.linalg.diag_part(nb)

    # Get the distance matrix
    distance = tf.sqrt(tf.maximum(tf.expand_dims(square_norm_na, 0)
                                  - 2 * tf.matmul(embed_masked, tf.transpose(embed_unmasked))
                                  + tf.expand_dims(square_norm_nb, 1), 0.0))

    # Tile the label array: each row contains the labels
    label_array = tf.expand_dims(labels, axis=0)
    label_array = tf.tile(label_array, [len(labels), 1])

    # Tile the label array: each row contains the label at the index of the current image, repeated
    diag_label = tf.expand_dims(labels, axis=-1)
    diag_label = tf.tile(diag_label, [1, len(labels)])

    # Get the max distance for each image
    max_dist = tf.reduce_max(distance, axis=0)
    max_dist = tf.expand_dims(max_dist, axis=-1)
    max_dist = tf.tile(max_dist, [1, len(max_dist)])

    diag = tf.linalg.diag_part(distance)

    zero_array = tf.zeros(shape=distance.shape)

    # Get the index of positive with largest distance
    find_max_positive = tf.where(label_array == diag_label, distance, zero_array)
    diag = tf.reduce_max(find_max_positive, axis=0)
    positive_indices = tf.math.argmax(find_max_positive, axis=0)

    dot_product = tf.where(label_array == diag_label, max_dist, distance)

    mins = tf.reduce_min(dot_product, axis=0)
    indices = tf.math.argmin(dot_product, axis=0)

    mask = mins <= diag
    none_array = tf.fill(tf.shape(indices), -1)
    none_array = tf.cast(none_array, tf.int64)

    negative_indices = tf.where(mask, indices, none_array)

    if chatty:
        print("Dist", distance.numpy())
        print("Label", label_array == diag_label)
        print("Find_max_positive", find_max_positive)
        print("Positive_indices", positive_indices)
        print("Diag", diag.numpy())
        print("Mins", mins.numpy())
        print("Negative", negative_indices.numpy())
        print("Positive", positive_indices.numpy())

    return negative_indices, positive_indices


class Dataset:

    def __init__(self, train_test_split=(0.8, 0.2), batch_size=32, base_path_str="../MaskedFaceGeneration"):

        if sum(train_test_split) != 1 or len(train_test_split) != 2:
            raise ValueError("Train/test split must be equal to 1")

        self.train_split = train_test_split[0]
        self.batch_size = batch_size

        self.basePath = pathlib.Path(base_path_str)

        folder_list = list(pathlib.Path(f"{str(self.basePath / 'LFW')}").glob("./*"))
        unmasked_examples = [list(folder.glob("*.jpg")) for folder in folder_list
                             if len(list(folder.glob("*.jpg"))) > 1]
        unmasked_examples = list(itertools.chain.from_iterable(unmasked_examples))

        folder_list = list(pathlib.Path(f"{str(self.basePath / 'LFW-masked')}").glob("./*"))
        masked_examples = [list(folder.glob("*.jpg")) for folder in folder_list
                           if len(list(folder.glob("*.jpg"))) > 1]
        masked_examples = list(itertools.chain.from_iterable(masked_examples))

        self.datasetUnmasked = unmasked_examples
        self.datasetMasked = masked_examples

        self.all_labels = []
        self.le = sklearn.preprocessing.LabelEncoder()

        self._masked_train_data = self._masked_test_data = self._unmasked_train_data = \
            self._unmasked_test_data = tf.data.Dataset

        self._masked_test_data_iter = self._masked_train_data_iter = self._unmasked_test_data_iter = \
            self._unmasked_train_data_iter = self._test_labels_iter = self._train_labels_iter = None

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
                    masked.append(str(masked_path.resolve(strict=True)))
                    unmasked.append(str(unmasked_path.resolve(strict=True)))
                    labels.append(label)
                    self.all_labels.add(label)

        self._masked_train_data = tf.data.Dataset.from_tensor_slices(masked_train)
        self._unmasked_train_data = tf.data.Dataset.from_tensor_slices(unmasked_train)

        self._masked_test_data = tf.data.Dataset.from_tensor_slices(masked_test)
        self._unmasked_test_data = tf.data.Dataset.from_tensor_slices(unmasked_test)

        test_train_sets = [self._masked_train_data, self._unmasked_train_data, self._masked_test_data,
                           self._unmasked_test_data]

        self.all_labels = list(self.all_labels)
        self.le.fit(self.all_labels)

        self._train_labels = tf.data.Dataset.from_tensor_slices(self.le.transform(train_labels))
        self._test_labels = tf.data.Dataset.from_tensor_slices(self.le.transform(test_labels))
        test_train_labels = [self._train_labels, self._test_labels]

        return test_train_sets, test_train_labels

    def get_data(self, masked=True, train=True):

        if masked and train:
            return self._masked_train_data
        if masked:
            return self._masked_test_data
        if train:
            return self._unmasked_train_data

        return self._unmasked_test_data

    def get_labels(self, train=True):

        if train:
            return self._train_labels

        return self._test_labels

    @staticmethod
    def process_path(file_path):
        try:
            image_raw = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image_raw, channels=3)
            img_conv = tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False)
            img_resize = tf.image.resize(img_conv, size=(constants.IMAGE_SIZE, constants.IMAGE_SIZE))
            ind_img = tf.reshape(img_resize, shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

            return ind_img
        except TypeError:
            print(file_path)

    def decode_label(self, label: tf.Tensor):
        label = tf.argmax(tf.cast(label == self.all_labels, tf.int32))
        return label

    def next_triplet(self):
        return next(self._masked_epoch_data), \
               next(self._unmasked_same_epoch_data), \
               next(self._unmasked_diff_epoch_data), \
               next(self._negative_epoch_data)

    def extract_sets(self, negative_indices, positive_indices, num_el, train=True):

        if train:
            masked_set = self._masked_train_data
            unmasked_set = self._unmasked_train_data
        else:
            masked_set = self._masked_test_data
            unmasked_set = self._unmasked_test_data

        masked_list_enum = list(masked_set.take(num_el).as_numpy_iterator())
        unmasked_list_enum = list(unmasked_set.take(num_el).as_numpy_iterator())

        masked_epoch_set = tf.data.Dataset.from_tensor_slices(
            [str(path, 'ascii') for i, path in enumerate(masked_list_enum) if negative_indices[i] != -1])
        unmasked_epoch_same_set = tf.data.Dataset.from_tensor_slices(
            [str(path, 'ascii') for i, path in enumerate(unmasked_list_enum) if negative_indices[i] != -1])

        unmasked_list = np.array([unmasked_list_enum[i] for idx, i in enumerate(positive_indices) if negative_indices[idx] != -1])
        unmasked_epoch_diff_set = tf.data.Dataset.from_tensor_slices([str(path, 'ascii') for path in unmasked_list])

        negatives_list = np.array([unmasked_list_enum[i] for i in negative_indices if i != -1])
        negatives_epoch_set = tf.data.Dataset.from_tensor_slices([str(path, 'ascii') for path in negatives_list])

        masked_set = masked_epoch_set.map(Dataset.process_path)
        unmasked_same_set = unmasked_epoch_same_set.map(Dataset.process_path)
        unmasked_diff_set = unmasked_epoch_diff_set.map(Dataset.process_path)
        negatives_set = negatives_epoch_set.map(Dataset.process_path)

        assert len(unmasked_diff_set) == len(unmasked_same_set) == len(masked_set) == len(negatives_set)

        self._masked_epoch_data = iter(masked_set)
        self._unmasked_same_epoch_data = iter(unmasked_same_set)
        self._unmasked_diff_epoch_data = iter(unmasked_diff_set)
        self._negative_epoch_data = iter(negatives_set)

        return len(masked_set)
