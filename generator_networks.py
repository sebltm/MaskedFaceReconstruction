import tensorflow as tf
import tensorflow.keras.activations
import tensorflow.keras.layers as layers

import constants


class ClassicGenerator(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1_1 = layers.Conv2D(filters=64, kernel_size=4, strides=1)
        self.conv1_2 = layers.Conv2D(filters=64, kernel_size=4, strides=1)
        self.max_pool1 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2_1 = layers.Conv2D(filters=128, kernel_size=4, strides=1)
        self.conv2_2 = layers.Conv2D(filters=128, kernel_size=4, strides=1)
        self.max_pool2 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv3_1 = layers.Conv2D(filters=256, kernel_size=4, strides=1)
        self.conv3_2 = layers.Conv2D(filters=256, kernel_size=4, strides=1)
        self.conv3_3 = layers.Conv2D(filters=256, kernel_size=4, strides=1)
        self.conv3_4 = layers.Conv2D(filters=256, kernel_size=4, strides=1)
        self.max_pool3 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv4_1 = layers.Conv2D(filters=512, kernel_size=4, strides=1)
        self.conv4_2 = layers.Conv2D(filters=512, kernel_size=4, strides=1)
        self.conv4_3 = layers.Conv2D(filters=512, kernel_size=4, strides=1)
        self.conv4_4 = layers.Conv2D(filters=512, kernel_size=4, strides=1)
        self.max_pool4 = layers.MaxPool2D(pool_size=2, strides=2)

        ### DECODER ###

        self.dec_upsamp1 = layers.UpSampling2D(size=(2, 2))
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

    def model(self):
        x = layers.Input(shape=(256, 256, 3))
        return tf.keras.Model(inputs=x, outputs=self.dec_conv5_2)

    def call(self, inputs, training=False, mask=None):
        outputs = self.conv1_1(inputs)
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

        ### DECODER ###
        outputs = self.dec_upsamp1(outputs)
        outputs = self.dec_conv1_1(outputs)
        outputs = self.dec_conv1_2(outputs)
        outputs = self.dec_conv1_3(outputs)
        outputs = self.dec_conv1_4(outputs)
        outputs = self.batch_norm1(outputs)

        outputs = self.dec_upsamp2(outputs)
        outputs = self.dec_conv2_1(outputs)
        outputs = self.dec_conv2_2(outputs)
        outputs = self.dec_conv2_3(outputs)
        outputs = self.dec_conv2_4(outputs)
        outputs = self.batch_norm2(outputs)

        outputs = self.dec_upsamp3(outputs)
        outputs = self.dec_conv3_1(outputs)
        outputs = self.dec_conv3_2(outputs)
        outputs = self.batch_norm3(outputs)

        outputs = self.dec_upsamp4(outputs)
        outputs = self.dec_conv4_1(outputs)
        outputs = self.dec_conv4_2(outputs)
        outputs = self.batch_norm4(outputs)

        outputs = self.dec_conv5_1(outputs)
        outputs = self.dec_conv5_2(outputs)

        return outputs


class ResNetGenerator(tf.keras.Model):

    def __init__(self, *args, outputs_channels=3, initializer=tf.random_normal_initializer(0., 0.02), **kwargs):
        super().__init__(*args, **kwargs)

        self.input_layer = layers.Input(shape=(constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))

        self.downstack = [
            self.Downsample(64, 3, apply_batchnorm=False),
            self.Downsample(128, 3),
            self.Downsample(256, 3),
            self.Downsample(512, 3),
            self.Downsample(512, 3),
            self.Downsample(512, 3),
            self.Downsample(512, 3),
            self.Downsample(512, 3)
        ]

        self.upstack = [
            self.Upsample(512, 3, apply_dropout=True),
            self.Upsample(512, 3, apply_dropout=True),
            self.Upsample(512, 3, apply_dropout=True),
            self.Upsample(512, 3),
            self.Upsample(256, 3),
            self.Upsample(128, 3),
            self.Upsample(64, 3)
        ]

        self.last = layers.Conv2DTranspose(outputs_channels, 4, strides=2, padding='same', activation='relu')

        self.out = self.call(self.input_layer)

    def model(self):
        x = layers.Input(shape=(256, 256, 3))
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    def call(self, inputs, training=None, mask=None):
        output = inputs

        skips = []
        for down in self.downstack:
            output = down(output)
            skips.append(output)

        skips = reversed(skips[:-1])

        for up, skip in zip(self.upstack, skips):
            output = up(output)
            output = layers.add([skip, output])
            output = tensorflow.keras.activations.relu(output)

        output = self.last(output)

        return output

    class Downsample(tf.keras.layers.Layer):

        def __init__(self, filters, size, padding='same', strides=2,
                     apply_batchnorm=True, initializer=tf.random_normal_initializer(0., 0.02)):
            super().__init__()

            self.initializer = initializer

            self.conv = layers.Conv2D(filters, size, strides=strides, padding=padding, use_bias=False)

            self.apply_batchnorm = apply_batchnorm

            self.batch_norm = layers.BatchNormalization()

        def call(self, inputs, *args, **kwargs):
            output = self.conv(inputs)

            return output

    class Upsample(tf.keras.layers.Layer):

        def __init__(self, filters, size, padding='same', strides=2,
                     apply_dropout=False):
            super().__init__()

            self.conv = layers.Conv2DTranspose(filters, size, strides=strides, padding=padding, use_bias=False)

            self.batch_norm = layers.BatchNormalization()

            self.apply_dropout = apply_dropout

            self.dropout = layers.Dropout(0.5)


        def call(self, inputs, *args, **kwargs):
            outputs = self.conv(inputs)

            return outputs


class Reconstruct(tf.keras.Model):

    def __init__(self, faceNet, output_size, dims, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_size = output_size
        self.dims = dims

        self.faceNet = faceNet

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

    def call(self, inputs, training=None, mask=None):

        output = self.faceNet(inputs)
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


class CapsGenerator(tf.keras.Model):

    def __init__(self, primary_caps):
        self.primary_caps = primary_caps


if __name__ == "__main__":
    classic_gen = ClassicGenerator()
    classic_gen.build(input_shape=(1, constants.IMAGE_SIZE, constants.IMAGE_SIZE, 3))
    classic_gen.summary()
