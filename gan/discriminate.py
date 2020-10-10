# %% Import libraries
import tensorflow as tf
from tensorflow import keras

# %% Discriminator model D
class Discriminator(tf.Module):
    def __init__(self, input_shape: tuple) -> None:
        super().__init__()
        # Network layers
        self.layers = [
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(units=512),
            keras.layers.ELU(),
            keras.layers.Dense(units=256),
            keras.layers.ELU(),
            keras.layers.Dense(units=1),
            keras.layers.Activation(tf.nn.sigmoid),
        ]

    @tf.Module.with_name_scope
    def __call__(self, input_data, training=False):
        output_data = input_data
        for layer in self.layers:
            output_data = layer(output_data)
        return output_data

    # Stochastic Gradient Descent used, as mentioned in the original algorithm
    @staticmethod
    def optimizer(learning_rate: float, momentum: float=0.0):
        return tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    # Asc_Loss(D) = mean(log(D(x_i))+log(1-D(G(z_i))))   [Algorithm 1 from GAN Paper]
    # Desc_Loss(D) = -mean(log(D(x_i))+log(1-D(G(z_i))))
    @staticmethod
    def loss(trained_ouput, generated_output) -> tf.Tensor:
        # trained_output = D(x_i)
        # generated_output = D(G(z_i)), fed directly from G
        # x_i sampled from training distribution
        # z_i sampled from generator distribution
        loss_i = tf.math.log(trained_ouput) + tf.math.log(1 - generated_output)
        loss = -tf.math.reduce_mean(loss_i)
        return loss

# %%

# %%
