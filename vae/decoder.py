# %% Import libraries
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

# %%
class Decoder(tf.Module):
    def __init__(self, latent_dimension: int, output_shape: tuple) -> None:
        super().__init__()
        self.latent_dimension = latent_dimension
        self.layers = [
            keras.layers.InputLayer(input_shape=(latent_dimension,)),
            keras.layers.Dense(units=256),
            keras.layers.Activation(tf.nn.tanh),
            keras.layers.Dense(units=np.prod(output_shape)),
            keras.layers.Activation(tf.nn.sigmoid),
            keras.layers.Reshape(target_shape=output_shape),
        ]

    @tf.Module.with_name_scope
    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        output_data = input_data
        for layer in self.layers:
            output_data = layer(output_data)
        return output_data
