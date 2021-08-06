import tensorflow as tf


def squash(tensor: tf.Tensor, axis=-2):
    s_squared_norm = tf.reduce_sum(tf.square(tensor), axis=axis, keepdims=True)

    safe_norm = tf.sqrt(s_squared_norm + 1e-7)
    scale = s_squared_norm / (1 + s_squared_norm)
    u_vec = tensor / safe_norm
    return scale * u_vec


class CapsLayer(tf.keras.layers.Layer):

    def __init__(self, num_caps, dim_caps, hist_writer=None, routings=3, init='glorot_uniform', **kwargs):
        super().__init__(**kwargs)

        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.routings = routings
        self.kernel_init = tf.keras.initializers.get(init)
        self.squash_agreement = tf.keras.layers.Lambda(squash, arguments={"axis": -1}, dtype='float32')

        self.batch_size = None
        self.input_num_caps = None
        self.input_dim_caps = None
        self.W = None

        self.reshape_input = None

        self.hist_writer = hist_writer
        self.step = 0

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.input_num_caps = input_shape[1]
        self.input_dim_caps = input_shape[2]

        # Shape = (219024, 10, 8, 8)
        self.W = self.add_weight(shape=[self.input_num_caps, self.num_caps,
                                        self.dim_caps, self.input_dim_caps],
                                 initializer=self.kernel_init, name='weight_matrix', trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        # b = batch
        # i = input_num_caps
        # j = num_caps
        # k = input_dim_caps
        # l = dim_caps

        # W_shape = (input_num_caps, num_caps, dim_caps, input_dim_caps) = (i, j, l, k)
        # inputs_shape = (batch, input_num_caps, input_dim_caps) = (b, i, k)
        # output_shape = (batch, input_num_caps, num_caps, dim_caps) = (b, i, j, l)
        uji = tf.einsum('ijlk,bik->bijl', self.W, inputs, name="predict_caps2")

        if self.hist_writer:
            with self.hist_writer.as_default():
                tf.summary.histogram("uji", uji, step=self.step)

        # b_shape = (batch, input_num_caps, num_caps) = (batch, 219024, 10, 1)
        b = tf.zeros(shape=[1, self.input_num_caps, self.num_caps, 1])
        vj = None
        for i in range(self.routings):
            cj = tf.nn.softmax(b, axis=-1)

            # s_shape = (batch, 1, 10, 8)
            s_pred = tf.multiply(cj, uji)
            s = tf.reduce_sum(s_pred, axis=1, keepdims=True)
            if self.hist_writer:
                with self.hist_writer.as_default():
                    tf.summary.histogram("s", s, step=self.step)

            vj = self.squash_agreement(s)
            if self.hist_writer:
                with self.hist_writer.as_default():
                    tf.summary.histogram("vj", vj, step=self.step)

            # uji_shape = (batch, 123008, 10, 8) = (b, i, j, k)
            # vj_shape = (batch, 1, 10, 8) = (b, l, j, k)
            # agreement_output = (batch, 123008, 10, 1) = (b, i, j, l)
            agreement = tf.einsum('bijk,bljk->bijl', uji, vj)
            # agreement = tf.matmul(uji, vj_tiled, transpose_a=True)
            if self.hist_writer:
                with self.hist_writer.as_default():
                    tf.summary.histogram("agreement", agreement, step=self.step)

            b += agreement

        return vj

    def reset(self):
        self.step = 0


class PrimaryCaps(tf.keras.layers.Layer):

    def __init__(self, dim_capsule=8, n_channels=8, kernel=9, strides=2, padding='valid'):
        super(PrimaryCaps, self).__init__()

        # Input shape = (batch, 242, 242, 256)
        self.conv = tf.keras.layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel, strides=strides,
                                           padding=padding, name="primary_conv")  # Output shape = (batch, 117, 117, dim * channel)
        self.reshape = tf.keras.layers.Reshape(target_shape=[-1, dim_capsule], name="reshape_primary")  # Output shape = (batch, num_caps, dim)
        self.squash = tf.keras.layers.Lambda(squash, name="squash_primary")  # Output shape = (batch, num_caps, dim)

    def call(self, inputs, **kwargs):

        output = self.conv(inputs)
        output = self.reshape(output)

        return self.squash(output)
