import os
import tensorflow as tf
import tensorflow_addons as tfa


class DownsampleLayer(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=4):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=2,
            padding="same")
        self.normalisation = tfa.layers.InstanceNormalization(
            axis=-1,
            center=False,
            scale=False)

    def __call__(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.normalisation(x, training=training)
        return tf.keras.layers.Activation("relu")(x)


class UpsampleLayer(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=4):
        super().__init__()
        self.upsampling = tf.keras.layers.UpSampling2D(size=2)
        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=1,
            padding="same")
        self.normalisation = tfa.layers.InstanceNormalization(
            axis=-1,
            center=False,
            scale=False)
        self.concat = tf.keras.layers.Concatenate()

    def __call__(self, inputs, training=True):
        assert len(inputs) == 2
        input, skip_input = inputs

        x = self.upsampling(input)
        x = self.conv(x)
        x = self.normalisation(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = self.concat([x, skip_input])
        return x


class UNet(tf.keras.Model):

    def __init__(self, filters=[32, 64, 128, 256], channels=3):
        super().__init__()

        assert len(filters) == 4

        self.d1 = DownsampleLayer(filters[0])
        self.d2 = DownsampleLayer(filters[1])
        self.d3 = DownsampleLayer(filters[2])
        self.d4 = DownsampleLayer(filters[3])

        self.u1 = UpsampleLayer(filters[2])
        self.u2 = UpsampleLayer(filters[1])
        self.u3 = UpsampleLayer(filters[0])
        self.u4 = tf.keras.layers.UpSampling2D(size=2)

        self.out = tf.keras.layers.Conv2D(
            channels,
            kernel_size=4,
            strides=1,
            padding="same",
            activation="tanh"
        )

    def __call__(self, inputs, training=True):
        dx1 = self.d1(inputs, training=training)
        dx2 = self.d2(dx1, training=training)
        dx3 = self.d3(dx2, training=training)
        x = self.d4(dx3, training=training)

        x = self.u1([x, dx3], training=training)
        x = self.u2([x, dx2], training=training)
        x = self.u3([x, dx1], training=training)
        x = self.u4(x)

        return self.out(x)
