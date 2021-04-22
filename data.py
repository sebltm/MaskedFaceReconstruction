import pathlib
import tensorflow as tf
import sklearn.model_selection
import numpy as np
from tqdm import tqdm
import network
import math
import datetime


def process_path(file_path: pathlib.WindowsPath):

    img_file = tf.io.read_file(str(file_path))
    image = tf.image.decode_jpeg(img_file, channels=3)
    img_conv = tf.image.convert_image_dtype(image, tf.float32)
    return img_conv


class Dataset(tf.keras.utils.Sequence):

    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.__data__ = []

    def __len__(self):
        return math.ceil(len(self.__data__) / self.batch_size)

    def __getitem__(self, index):
        batch = self.__data__[index * self.batch_size:(index + 1) * self.batch_size]

        xs = []
        ys = []
        for idx, next_el in enumerate(batch):
            unmasked_img_real = process_path(next_el[0])
            masked_img_real = process_path(next_el[1])
            masked_img_fake = process_path(next_el[2])

            x = [unmasked_img_real, masked_img_real, masked_img_fake]
            y = np.zeros(shape=(10, 8))

            xs.append(x)
            ys.append(y)

        xs = tf.convert_to_tensor(xs)
        ys = tf.convert_to_tensor(ys)
        return tuple([xs, ys])


class TrainSet(Dataset):

    def __init__(self, train_test_split=(0.8, 0.2), batch_size=32, base_path_str="../MaskedFaceGeneration"):
        super().__init__(batch_size=batch_size)

        if sum(train_test_split) != 1 or len(train_test_split) != 2:
            raise ValueError("Train test split must be equal to 1")

        self.train_split = train_test_split[0]

        self.basePath = pathlib.Path(base_path_str)
        self.datasetUnmasked = list(pathlib.Path(f"{str(self.basePath / 'LFW')}").glob("./**/*.jpg"))
        self.datasetMasked = list(pathlib.Path(f"{str(self.basePath / 'LFW-masked')}").glob("./**/*.jpg"))

        self.__test__ = []
        self.process_split()

    def process_split(self):
        self.__data__ = []
        self.__test__ = []

        masked_train = []
        masked_test = []

        unmasked_train_els, unmasked_test_els = sklearn.model_selection.train_test_split(self.datasetUnmasked,
                                                                                         train_size=0.8)

        for zipped in tqdm(zip([unmasked_train_els, unmasked_test_els],
                               [self.__data__, self.__test__],
                               [masked_train, masked_test])):

            train_els = zipped[0]
            final_set = zipped[1]
            masked_set = zipped[2]

            for idx, unmasked_path in enumerate(train_els):
                masked_path = list(unmasked_path.parts)
                masked_path[2] = 'LFW-masked'
                masked_path = pathlib.Path("/".join(masked_path))

                if masked_path.exists():
                    masked_set.append(masked_path)
                    final_set.append([unmasked_path, masked_path])

        masked_fakes = np.random.choice(masked_train, size=len(self.__data__), replace=True)
        self.__data__ = [x + [y] for x, y in zip(self.__data__, masked_fakes)]

        masked_fakes = np.random.choice(masked_test, size=len(self.__test__), replace=True)
        self.__test__ = [x + [y] for x, y in zip(self.__test__, masked_fakes)]


class TestSet(Dataset):

    def __init__(self, dataset: TrainSet, batch_size=32):
        super().__init__(batch_size=batch_size)

        self.dataset = dataset
        self.__data__ = dataset.__test__


if __name__ == "__main__":

    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    batch_size = 1

    train_set = TrainSet(batch_size=batch_size)
    test_set = TestSet(train_set, batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    loss_metric = tf.keras.metrics.Mean()

    net = network.SiameseCaps(output_size=10)
    net.build(input_shape=(None, 3, 250, 250, 3))
    net.summary()

    loss_fn = net.TripletLoss()
    net.compile(loss=loss_fn, optimizer='adam')

    training_history = net.fit(
        train_set,
        batch_size=batch_size,
        steps_per_epoch=50,
        epochs=5,
        verbose=True,
        validation_data=test_set,
        callbacks=[tensorboard_callback]
    )
