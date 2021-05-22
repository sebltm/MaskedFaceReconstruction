import capsules
import tensorflow as tf
import tensorflow.keras.layers as layers

import data


class VGG16(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reshape = layers.Reshape(input_shape=(250, 250, 3), target_shape=(250, 250, 3))

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
        self.fc4 = layers.Dense(units=80, activation='relu')

    def call(self, inputs, training=None, mask=None):
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

    def __init__(self, output_size, filters=256, kernel=9, strides_conv=1, strides_caps=2, routings=3):
        super(FaceCapsNet, self).__init__()

        self.kernel = kernel
        self.strides_caps = strides_caps

        # Output = (batch, 250, 250, 3)
        self.reshape = layers.Reshape(target_shape=(250, 250, 3))

        # Output shape = (batch, 242, 242, 256)
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides_conv, padding='valid',
                                  activation='relu')

        # Output shape = (batch, 219024, 8)
        self.primary_caps = capsules.PrimaryCaps(dim_capsule=8, n_channels=16, kernel=self.kernel,
                                                 strides=self.strides_caps, padding='valid')

        # Output shape =
        self.face_caps = capsules.CapsLayer(num_caps=output_size, dim_caps=8, routings=routings)

        self.primary_caps.build(input_shape=(None, 242, 242, 256))
        self.face_caps.build(input_shape=(None, 219024, 8))

    def call(self, inputs, training=None, mask=None):
        inputs = self.reshape(inputs)
        x1 = self.conv(inputs)
        x1 = self.primary_caps(x1)
        x1 = self.face_caps(x1)
        return x1

    def train(self, model: tf.keras.Model, data, args):
        model.compile(optimizer=tf.optimizers.Adam(lr=args.lr),
                      loss=[self.MyLoss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})


class SiameseCaps(tf.keras.models.Model):

    def __init__(self, output_size, filters=256, kernel=9, strides_conv=1, strides_caps=2, routings=3, caps=True, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if caps:
            self.faceNet = FaceCapsNet(output_size=output_size, filters=filters, kernel=kernel,
                                       strides_conv=strides_conv,
                                       strides_caps=strides_caps, routings=routings)
        else:
            self.faceNet = VGG16()

        self.faceNet.build(input_shape=(None, 250, 250, 3))
        self.faceNet.summary()

        self.reshape = tf.keras.layers.Reshape(target_shape=(3, 250, 250, 3))
        self.lambda_split = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))
        self.s1 = self.faceNet
        self.s2 = self.faceNet
        self.s3 = self.faceNet

        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.out_shape = tf.keras.layers.Reshape(target_shape=(3, 10, 8))

        self.lambda_split.build(input_shape=(3, 250, 250, 3))

    def call(self, inputs, training=None, mask=None):
        inputs = self.reshape(inputs)
        an, po, neg = self.lambda_split(inputs)

        anchor = self.s1(an)
        positive = self.s2(po)
        negative = self.s3(neg)

        out = self.concat([anchor, positive, negative])
        out = self.out_shape(out)
        return out

    # def train(self, train_set: data.Dataset, epochs=15):
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    #
    #     loss_metric = tf.keras.metrics.Mean()
    #
    #     self.build(input_shape=(None, 3, 250, 250, 3))
    #     self.summary()
    #
    #     loss_fn = self.TripletLoss()
    #     self.compile(loss=loss_fn, optimizer='adam')
    #
    #     for epoch in range(epochs):
    #         embeddings = self.faceCaps.predict(train_set)
    #
    #         # GET HARD TRIPLETS
    #         triplet_ds = [embeddings]
    #         for step, (x_batch_train, y_batch_train) in enumerate(triplet_ds):
    #             with tf.GradientTape() as tape:
    #
    #                 logits = self(x_batch_train, training=True)
    #
    #                 loss_value = loss_fn(y_batch_train, logits)
    #
    #             grads = tape.gradient(loss_value, self.trainable_weights)
    #
    #             optimizer.apply_gradients(zip(grads, self.trainable_weights))

    class TripletLoss(tf.keras.losses.Loss):

        def __init__(self, alpha=10e-5):
            super().__init__()

            self.alpha = alpha

        def call(self, y_true, y_pred):
            batch_size = tf.shape(y_pred)[0]

            anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)

            anchor = tf.reshape(anchor, shape=(batch_size, -1))
            positive = tf.reshape(positive, shape=(batch_size, -1))
            negative = tf.reshape(negative, shape=(batch_size, -1))

            # print(anchor)
            # print(positive)
            # print(negative)
            #
            # print()
            # print(anchor - positive)
            # print(anchor - negative)

            positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

            # print("Positive", positive_dist)
            # print("Negative", negative_dist)
            #
            # print("\n")

            loss = tf.maximum(positive_dist - negative_dist + self.alpha, 0.)

            # print("Loss", loss)

            return loss
