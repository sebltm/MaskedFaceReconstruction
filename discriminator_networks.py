import tensorflow as tf
import tensorflow.keras.layers as layers

import capsules, constants, generator_networks as G


class VGG16(tf.keras.Model):

    def __init__(self, output_size=16, dims=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = layers.Input(shape=(constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

        self.reshape = layers.Reshape(input_shape=(constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3), target_shape=(constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

        self.conv1_1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv1_2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')
        self.max_pool1 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2_1 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv2_2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')
        self.max_pool2 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv3_1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3_2 = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3_3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv3_4 = layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding='same')
        self.max_pool3 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv4_1 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv4_2 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv4_3 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv4_4 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
        self.max_pool4 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv5_1 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv5_2 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv5_3 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
        self.conv5_4 = layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding='same')
        self.max_pool5 = layers.MaxPool2D(pool_size=2, strides=2)

        self.flatten = layers.Flatten()

        self.fc1 = layers.Dense(units=4096, activation='relu')
        self.fc2 = layers.Dense(units=4096, activation='relu')
        self.fc3 = layers.Dense(units=1000, activation='relu')
        self.fc4 = layers.Dense(units=output_size * dims, activation='relu', dtype='float32')

        self.out = self.call(self.input_layer)

    def call(self, inputs, training=False, mask=None):
        outputs = self.reshape(inputs)

        outputs = self.conv1_1(outputs)
        outputs = self.conv1_2(outputs)
        outputs = self.max_pool1(outputs)

        outputs = self.conv2_1(outputs)
        outputs = self.conv2_2(outputs)
        outputs = self.max_pool2(outputs)

        outputs = self.conv3_1(outputs)
        outputs = self.conv3_2(outputs)
        outputs = self.conv3_3(outputs)
        outputs = self.conv3_4(outputs)
        outputs = self.max_pool3(outputs)

        outputs = self.conv4_1(outputs)
        outputs = self.conv4_2(outputs)
        outputs = self.conv4_3(outputs)
        outputs = self.conv4_4(outputs)
        outputs = self.max_pool4(outputs)

        outputs = self.conv5_1(outputs)
        outputs = self.conv5_2(outputs)
        outputs = self.conv5_3(outputs)
        outputs = self.conv5_4(outputs)
        outputs = self.max_pool5(outputs)

        outputs = self.flatten(outputs)

        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        outputs = self.fc3(outputs)
        outputs = self.fc4(outputs)

        return outputs


class FaceCapsNet(tf.keras.Model):

    def __init__(self, mid_size, output_size, dims=8, filters=256, kernel=14, strides_conv=1, strides_caps=3, routings=3,
                 hist_writer=None, reconstruct=False):
        super(FaceCapsNet, self).__init__()

        self.input_layer = layers.Input(shape=(constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

        self.kernel = kernel
        self.strides_caps = strides_caps
        self.reconstruct = reconstruct
        self.output_size = output_size
        self.dims = dims

        # Output = (batch, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3)
        self.reshape = layers.Reshape(target_shape=(constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

        # # Output shape = (batch, 242, 242, 256)
        # self.conv = layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides_conv, padding='valid',
        #                           activation='relu')

        # Output shape = (batch, 219024, 8)
        self.primary_caps = capsules.PrimaryCaps(dim_capsule=8, n_channels=mid_size, kernel=self.kernel,
                                                 strides=self.strides_caps, padding='valid')

        # Output shape =
        self.face_caps = capsules.CapsLayer(num_caps=output_size, dim_caps=dims, routings=routings, hist_writer=hist_writer, dtype='float32')

        prim_caps_out = self.primary_caps.compute_output_shape(input_shape=(None, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

        self.primary_caps.build(input_shape=(None, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

        self.face_caps.build(input_shape=prim_caps_out)

        if self.reconstruct:
            # self.dec_upsamp1 = layers.UpSampling2D(size=(2, 2))
            self.dec_conv1_1 = layers.Conv2DTranspose(filters=512, kernel_size=5, strides=1)
            self.dec_conv1_2 = layers.Conv2DTranspose(filters=512, kernel_size=4, strides=1)
            self.dec_conv1_3 = layers.Conv2DTranspose(filters=512, kernel_size=4, strides=1)
            self.dec_conv1_4 = layers.Conv2DTranspose(filters=512, kernel_size=4, strides=1)
            self.batch_norm1 = layers.BatchNormalization()

            self.dec_upsamp2 = layers.UpSampling2D(size=(2, 2))
            self.dec_conv2_1 = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=1)
            self.dec_conv2_2 = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=1)
            self.dec_conv2_3 = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=1)
            self.dec_conv2_4 = layers.Conv2DTranspose(filters=256, kernel_size=4, strides=1)
            self.batch_norm2 = layers.BatchNormalization()

            self.dec_upsamp3 = layers.UpSampling2D(size=(2, 2))
            self.dec_conv3_1 = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=1)
            self.dec_conv3_2 = layers.Conv2DTranspose(filters=128, kernel_size=4, strides=1)
            self.batch_norm3 = layers.BatchNormalization()

            self.dec_upsamp4 = layers.UpSampling2D(size=(2, 2))
            self.dec_conv4_1 = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1)
            self.dec_conv4_2 = layers.Conv2DTranspose(filters=64, kernel_size=4, strides=1)
            self.batch_norm4 = layers.BatchNormalization()

            self.dec_conv5_1 = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1)
            self.dec_conv5_2 = layers.Conv2DTranspose(filters=3, kernel_size=4, strides=1, dtype='float32')

        self.out = self.call(self.input_layer)

    def call(self, inputs, training=False, mask=None):
        output = self.reshape(inputs)
        output = self.primary_caps(output)
        output = self.face_caps(output)

        if self.reconstruct:
            output = tf.reshape(output, shape=(-1, self.output_size, self.dims, 1))

            output = self.dec_conv1_1(output)
            output = self.dec_conv1_2(output)
            output = self.dec_conv1_3(output)
            output = self.dec_conv1_4(output)
            output = self.batch_norm1(output)

            output = self.dec_upsamp2(output)
            output = self.dec_conv2_1(output)
            output = self.dec_conv2_2(output)
            output = self.dec_conv2_3(output)
            output = self.dec_conv2_4(output)
            output = self.batch_norm2(output)

            output = self.dec_upsamp3(output)
            output = self.dec_conv3_1(output)
            output = self.dec_conv3_2(output)
            output = self.batch_norm3(output)

            output = self.dec_upsamp4(output)
            output = self.dec_conv4_1(output)
            output = self.dec_conv4_2(output)
            output = self.batch_norm4(output)

            output = self.dec_conv5_1(output)
            output = self.dec_conv5_2(output)

        return output

    def model(self):
        x = layers.Input(shape=(256, 256, 3))
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    def train(self, model: tf.keras.Model, data, args):
        model.compile(optimizer=tf.optimizers.Adam(lr=args.lr),
                      loss=[self.MyLoss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})

    def reset(self):
        self.face_caps.reset()


class SiameseCaps(tf.keras.models.Model):

    def __init__(self, mid_size, output_size, dims=8, filters=256, kernel=9, strides_conv=1, strides_caps=2, routings=3,
                 caps=True, hist_writer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.caps = caps

        self.input_layer = layers.Input(shape=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

        if caps:
            self.faceNet = FaceCapsNet(mid_size=mid_size, output_size=output_size, dims=dims, filters=filters,
                                       kernel=kernel, strides_conv=strides_conv, strides_caps=strides_caps,
                                       routings=routings, hist_writer=hist_writer)
        else:
            self.faceNet = VGG16(output_size=output_size, dims=dims)

        self.faceNet.build(input_shape=(None, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        self.faceNet.summary()

        self.reshape = tf.keras.layers.Reshape(target_shape=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        self.lambda_split = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))
        self.s1 = self.faceNet
        self.s2 = self.faceNet
        self.s3 = self.faceNet

        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.out_shape = tf.keras.layers.Reshape(target_shape=(3, output_size, dims))

        self.lambda_split.build(input_shape=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=False, mask=None):
        inputs = self.reshape(inputs)
        an, po, neg = self.lambda_split(inputs)

        anchor = self.s1(an)
        positive = self.s2(po)
        negative = self.s3(neg)

        out = self.concat([anchor, positive, negative])
        out = self.out_shape(out)
        return out

    def model(self):
        x = layers.Input(shape=(3, 256, 256, 3))
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    def reset(self):
        if self.caps:
            self.faceNet.reset()

    class TripletLoss(tf.keras.losses.Loss):

        def __init__(self, alpha=10e-10, gen=False, scale=1):
            super().__init__()

            self.alpha = alpha
            self.gen = gen
            self.scale = scale

        def call(self, y_true, y_pred):
            batch_size = tf.shape(y_pred)[0]

            anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)

            anchor = tf.reshape(anchor, shape=(batch_size, -1))
            positive = tf.reshape(positive, shape=(batch_size, -1))
            negative = tf.reshape(negative, shape=(batch_size, -1))

            positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=1)
            negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=1)

            loss = positive_dist - negative_dist + self.alpha

            if not self.gen:
                loss = tf.keras.backend.relu(loss)

            return loss * self.scale

    class BEGANLoss(tf.keras.losses.Loss):

        def __init__(self, k=1, diversity_ratio=0.7, learning_rate=0.1, scale=1):
            super().__init__()

            self.k = k
            self.learning_rate = learning_rate
            self.diversity_ratio = diversity_ratio
            self.scale = scale

        def call(self, y, x):
            y_true, y_pred = tf.split(y, num_or_size_splits=2, axis=0)
            x_true, x_pred = tf.split(x, num_or_size_splits=2, axis=0)

            y_fac = tf.subtract(y_true, y_pred)
            x_fac = tf.subtract(x_true, x_pred)

            y_dist = tf.reduce_sum(tf.square(y_fac), axis=1)
            x_dist = tf.reduce_sum(tf.square(x_fac), axis=1)

            loss = y_dist + self.k * x_dist

            self.k = self.k + self.learning_rate * (self.diversity_ratio * y_dist - x_dist)

            return loss * self.scale

        def update_lr(self, learning_rate):
            self.learning_rate = learning_rate

    class PixelLoss(tf.keras.losses.Loss):

        def __init__(self, scale=1):
            super().__init__()
            self.scale = scale

        def call(self, y_true, y_pred):
            return tf.reduce_mean(tf.square((tf.subtract(y_true, y_pred)))) * self.scale