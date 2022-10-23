import tensorflow as tf

from typing import Literal, Tuple


class NDDNet:
    def __init__(
        self,
        n_classes: int,
        n_conv_blocks: int,
        n_mlp_layers: int,
        conv_channel_width: int,
        mlp_channel_width: int,
        kernel_size: int,
        mode: Literal["grf", "features", "combined"]
    ):
        self.n_classes = n_classes
        self.n_conv_blocks = n_conv_blocks
        self.n_mlp_layers = n_mlp_layers
        self.conv_channel_width = conv_channel_width
        self.mlp_channel_width = mlp_channel_width
        self.kernel_size = kernel_size
        self.mode = mode

    @staticmethod
    def conv_block_1D(
        x: tf.Tensor,
        filters: int,
        kernel_size: int,
        name: str
    ) -> tf.Tensor:
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
            name=name
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        return x

    @staticmethod
    def conv_mixer_block_1D(
        x: tf.Tensor,
        filters: int,
        kernel_size: int,
        name: str
    ) -> tf.Tensor:
        # Depthwise convolution.
        x_init = x
        x = tf.keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer="l2",
            name="d_wise_" + name
        )(x)
        # Residual.
        x = tf.keras.layers.Add()([NDDNet.activation_block(x), x_init])

        # Pointwise convolution.
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=1,
            kernel_regularizer="l2",
            name="p_wise_" + name
        )(x)
        x = NDDNet.activation_block(x)

        return x

    @staticmethod
    def conv_stem(x, filters: int, patch_size: int):
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=patch_size,
            strides=patch_size,
            kernel_regularizer="l2"
        )(x)
        return NDDNet.activation_block(x)

    @staticmethod
    def activation_block(x: tf.Tensor) -> tf.Tensor:
        x = tf.keras.layers.Activation("gelu")(x)
        return tf.keras.layers.BatchNormalization()(x)

    @staticmethod
    def conv_channel_1D(
        x: tf.Tensor,
        n_blocks: int,
        init_width: int,
        kernel_size: int,
        name: str
    ) -> tf.Tensor:
        for i in range(n_blocks):
            filters = int(init_width * (2 ** i))
            x = NDDNet.conv_block_1D(
                x=x,
                filters=filters,
                kernel=kernel_size,
                name=f"channel_{name}_conv_layer_{i}"
            )

            if x.shape[1] <= 2:
                x = tf.keras.layers.MaxPooling1D(
                    pool_size=1, strides=2, padding="valid")(x)
            else:
                x = tf.keras.layers.MaxPooling1D(
                    pool_size=2, strides=2, padding="valid")(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)

        return x

    @staticmethod
    def conv_mixer_channel_1D(
        x: tf.Tensor,
        n_blocks: int,
        width: int,
        kernel_size: int,
        patch_size: int,
        name: str
    ) -> tf.Tensor:

        x = NDDNet.conv_stem(x, width, patch_size)

        for i in range(n_blocks):
            x = NDDNet.conv_mixer_block_1D(
                x=x,
                filters=width,
                kernel_size=kernel_size,
                name=f"channel_{name}_conv_layer_{i}"
            )

        x = tf.keras.layers.GlobalAvgPool1D()(x)

        return x

    @staticmethod
    def mlp_channel_1D(
        x: tf.Tensor,
        n_layers: int,
        width: int,
        name: str
    ) -> tf.Tensor:
        x = tf.keras.layers.Flatten()(x)
        for i in range(n_layers):
            x = tf.keras.layers.Dense(
                units=width,
                activation="relu",
                kernel_initializer="he_normal",
                name=f"mlp_channel_{name}_layer_{i}"
            )(x)

        x = tf.keras.layers.Dense(32, activation="relu")(x)

        return x

    def get_model(
        self,
        n_grf_channels: int,
        n_feature_channels: int,
        grf_channel_shape: Tuple[int, int],
        feature_channel_shape: Tuple[int, int]
    ) -> tf.keras.Model:
        inputs = []
        grf_embeddings = []
        feature_embeddings = []

        for i in range(n_grf_channels):
            grf_input = tf.keras.layers.Input(shape=grf_channel_shape)

            # ... Regular CNN
            # grf_output_i = self.conv_channel_1D(
            #     x=grf_input_i,
            #     n_blocks=self.n_conv_blocks,
            #     init_width=self.conv_channel_width,
            #     kernel_size=self.kernel_size,
            #     name=f"{i}"
            # )

            # ... ConvMixer
            grf_output = self.conv_mixer_channel_1D(
                x=grf_input,
                n_blocks=self.n_conv_blocks,
                width=self.conv_channel_width,
                kernel_size=self.kernel_size,
                patch_size=11,
                name=f"{i}"
            )

            inputs.append(grf_input)
            grf_embeddings.append(grf_output)

        for i in range(n_feature_channels):
            feature_input = tf.keras.layers.Input(shape=feature_channel_shape)

            feature_output = self.mlp_channel_1D(
                x=feature_input,
                n_layers=self.n_mlp_layers,
                width=self.mlp_channel_width,
                name=f"{i}"
            )

            inputs.append(feature_input)
            feature_embeddings.append(feature_output)

        if self.mode == "grf":
            x = tf.keras.layers.concatenate(grf_embeddings, axis=-1)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Dense(
                16, activation="relu", kernel_regularizer="l2")(x)
            x = tf.keras.layers.Dropout(0.2)(x)

        elif self.mode == "feature":
            x = tf.keras.layers.concatenate(feature_embeddings, axis=-1)
            x = tf.keras.layers.Dropout(0.2)(x)

        elif self.mode == "combined":
            x = tf.keras.layers.concatenate(grf_embeddings, axis=-1)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(16, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.concatenate([x] + feature_embeddings, axis=-1)
            x = tf.keras.layers.Dropout(0.2)(x)

        else:
            raise ValueError("Mode not correct!")

        output = tf.keras.layers.Dense(self.n_classes, activation="softmax")(x)

        return tf.keras.models.Model(inputs, output)
