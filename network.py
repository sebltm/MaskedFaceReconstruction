import capsules
import tensorflow as tf
import tensorflow.keras.layers as layers


class FaceCapsNet(tf.keras.Model):

    def __init__(self, output_size, filters=256, kernel=9, strides_conv=1, strides_caps=2, routings=3):
        super(FaceCapsNet, self).__init__()

        self.kernel = kernel
        self.strides_caps = strides_caps

        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides_conv, padding='valid',
                                  activation='relu')
        self.primary_caps = capsules.PrimaryCaps(dim_capsule=8, n_channels=16, kernel=self.kernel,
                                                 strides=self.strides_caps, padding='valid')
        self.face_caps = capsules.CapsLayer(num_caps=output_size, dim_caps=8, routings=routings)

    def call(self, inputs, training=None, mask=None):

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

    def __init__(self, output_size, filters=256, kernel=9, strides_conv=1, strides_caps=2, routings=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        faceCaps = FaceCapsNet(output_size=output_size, filters=filters, kernel=kernel, strides_conv=strides_conv,
                               strides_caps=strides_caps, routings=routings)
        faceCaps.build(input_shape=(1, 250, 250, 3))
        faceCaps.summary()

        self.reshape = tf.keras.layers.Reshape(target_shape=(3, 250, 250, 3))
        self.lambda_split = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))
        self.s1 = faceCaps
        self.s2 = faceCaps
        self.s3 = faceCaps

        self.concat = tf.keras.layers.Concatenate(axis=0)

        self.lambda_split.build(input_shape=(3, 250, 250, 3))

    def call(self, inputs, training=None, mask=None):

        inputs = self.reshape(inputs)
        an, po, neg = self.lambda_split(inputs)

        anchor = self.s1(an)
        positive = self.s2(po)
        negative = self.s3(neg)

        return tf.expand_dims(self.concat([anchor, positive, negative]), axis=0)

    class TripletLoss(tf.keras.losses.Loss):

        def __init__(self, alpha=0.0001):
            super().__init__()

            self.alpha = alpha

        def call(self, y_true, y_pred):
            size_batch = tf.shape(y_pred)[0]

            anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)

            anchor = tf.reshape(anchor, shape=(size_batch, 10, 8))
            positive = tf.reshape(positive, shape=(size_batch, 10, 8))
            negative = tf.reshape(negative, shape=(size_batch, 10, 8))

            positive_dist = tf.norm(anchor - positive)
            negative_dist = tf.norm(anchor - negative)

            return tf.maximum(positive_dist - negative_dist + self.alpha, 0.)
