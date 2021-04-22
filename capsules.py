import tensorflow as tf


def squash(tensor: tf.Tensor):
    s_squared_norm = tf.keras.backend.sum(tf.square(tensor), axis=1, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return tf.multiply(scale, tensor)


class CapsLayer(tf.keras.layers.Layer):

    def __init__(self, num_caps, dim_caps, routings=3, init='glorot_uniform', **kwargs):
        super().__init__(**kwargs)

        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.routings = routings
        self.kernel_init = tf.keras.initializers.get(init)
        self.squash = tf.keras.layers.Lambda(squash)

        self.input_num_caps = None
        self.input_dim_caps = None
        self.W = None

    def build(self, input_shape):
        self.input_num_caps = input_shape[1]
        self.input_dim_caps = input_shape[2]

        self.W = self.add_weight(shape=[1, self.input_num_caps, self.num_caps,
                                        self.input_dim_caps, self.dim_caps],
                                 initializer=self.kernel_init)

        self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.reshape(inputs, (1, -1, self.input_dim_caps))
        inputs = tf.expand_dims(inputs, axis=-2)
        inputs = tf.expand_dims(inputs, axis=-1)

        uji = tf.matmul(self.W, inputs)
        uji = tf.squeeze(uji, [4])

        b = tf.zeros(shape=[tf.shape(inputs)[0], tf.shape(inputs)[1], self.num_caps, 1])
        vj = None
        for i in range(self.routings):
            cj = tf.nn.softmax(b, axis=-2)
            s = tf.reduce_sum(tf.multiply(cj, uji), axis=1, keepdims=True)
            vj = self.squash(s)

            agreement = tf.squeeze(
                tf.matmul(tf.expand_dims(uji, axis=-1), tf.expand_dims(vj, axis=-1), transpose_a=True), [4], name="agreement")
            b += agreement

        return vj

# def PrimaryCaps(inputs, dim_capsule, n_channels, kernel, strides, padding):
#     output = tf.keras.layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel, strides=strides,
#                                        padding=padding)(inputs)
#     output = tf.keras.layers.Reshape(target_shape=[-1, dim_capsule])(output)
#     output = tf.keras.layers.Lambda(squash)(output)
#     return output


class PrimaryCaps(tf.keras.layers.Layer):

    def __init__(self, dim_capsule, n_channels, kernel, strides, padding):
        super(PrimaryCaps, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel, strides=strides,
                                           padding=padding)
        self.reshape = tf.keras.layers.Reshape(target_shape=[-1, dim_capsule])
        self.squash = tf.keras.layers.Lambda(squash)

    def call(self, inputs, **kwargs):
        output = self.conv(inputs)
        output = self.reshape(output)

        return self.squash(output)
