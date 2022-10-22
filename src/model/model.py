import tensorflow as tf


def conv_block_1D(
    x: tf.Tensor,
    filters: int,
    kernel: int,
    name: str
) -> tf.Tensor:
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel,
        padding="same",
        kernel_initializer="he_normal",
        name=name
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    return x


def activation_block(x):
    x = tf.keras.layers.Activation("gelu")(x)
    return tf.keras.layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=patch_size,
        strides=patch_size,
        kernel_regularizer="l2"
    )(x)
    return activation_block(x)


def conv_mixer_block_1D(
    x: tf.Tensor,
    filters: int,
    kernel: int,
    name: str
) -> tf.Tensor:
    # Depthwise convolution.
    x_init = x
    x = tf.keras.layers.DepthwiseConv1D(
        kernel_size=kernel,
        padding="same",
        kernel_regularizer="l2"
    )(x)
    # Residual.
    x = tf.keras.layers.Add()([activation_block(x), x_init])

    # Pointwise convolution.
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=1,
        kernel_regularizer="l2"
    )(x)
    x = activation_block(x)

    return x


def conv_mixer_channel_1D(
    x: tf.Tensor,
    n_blocks: int,
    init_width: int,
    kernel_size: int,
    patch_size: int,
    name: str
) -> tf.Tensor:

    x = conv_stem(x, init_width, patch_size)

    for i in range(n_blocks):
        x = conv_mixer_block_1D(
            x=x,
            filters=init_width,
            kernel=kernel_size,
            name=f"channel_{name}_conv_layer_{i}"
        )

    x = tf.keras.layers.GlobalAvgPool1D()(x)

    return x


def conv_channel_1D(
    x: tf.Tensor,
    n_blocks: int,
    init_width: int,
    kernel_size: int,
    name: str
) -> tf.Tensor:
    for i in range(n_blocks):
        filters = int(init_width * (2 ** i))
        x = conv_block_1D(
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
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    return x


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


def get_stacked_model(
    n_conv_channels: int,
    n_mlp_channels: int,
    n_blocks_conv: int,
    n_layers_mlp: int,
    conv_channel_width: int,
    mlp_channel_width: int,
    kernel_size: int,
    conv_channel_input_dim: tuple,
    mlp_channel_input_dim: tuple,
    mode: str
):
    inputs = []
    conv_channels = []
    mlp_channels = []

    for i in range(n_conv_channels):
        conv_input_i = tf.keras.layers.Input(shape=conv_channel_input_dim)

        # conv_channel_i = conv_channel_1D(
        #     x=conv_input_i,
        #     n_blocks=n_blocks_conv,
        #     init_width=conv_channel_width,
        #     kernel_size=kernel_size,
        #     name=f"{i}"
        # )

        conv_channel_i = conv_mixer_channel_1D(
            x=conv_input_i,
            n_blocks=n_blocks_conv,
            init_width=conv_channel_width,
            kernel_size=kernel_size,
            patch_size=11,
            name=f"{i}"
        )

        inputs.append(conv_input_i)
        conv_channels.append(conv_channel_i)

    for i in range(n_mlp_channels):
        mlp_input_i = tf.keras.layers.Input(shape=mlp_channel_input_dim)
        mlp_channel_i = mlp_channel_1D(
            x=mlp_input_i,
            n_layers=n_layers_mlp,
            width=mlp_channel_width,
            name=f"{i}"
        )
        inputs.append(mlp_input_i)
        mlp_channels.append(mlp_channel_i)

    if mode == "grf":
        x = tf.keras.layers.concatenate(conv_channels, axis=-1)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(
            16, activation="relu", kernel_regularizer="l2")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    elif mode == "gait":
        x = tf.keras.layers.concatenate(mlp_channels, axis=-1)
        x = tf.keras.layers.Dropout(0.2)(x)

    elif mode == "combined":
        x = tf.keras.layers.concatenate(conv_channels, axis=-1)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.concatenate([x] + mlp_channels, axis=-1)
        x = tf.keras.layers.Dropout(0.2)(x)

    else:
        raise ValueError("Mode not correct!")

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs, output)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model
