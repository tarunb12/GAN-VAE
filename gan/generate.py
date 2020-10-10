# %% Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras


# %% Generator model G
class Generator(tf.Module):
    def __init__(self, noise_shape: tuple, output_shape: tuple) -> None:
        super().__init__()
        # Network layers
        self.layers = [
            keras.layers.InputLayer(input_shape=noise_shape),
            keras.layers.Dense(units=128),
            keras.layers.ReLU(),
            keras.layers.Dense(units=256),
            keras.layers.ReLU(),
            keras.layers.Dense(units=512),
            keras.layers.ReLU(),
            keras.layers.Dense(units=np.prod(output_shape)),
            keras.layers.ReLU(),
            keras.layers.Reshape(target_shape=output_shape),
            keras.layers.Activation(tf.nn.sigmoid),
        ]

    @tf.Module.with_name_scope
    def __call__(self, input_data, training=False) -> tf.Tensor:
        output_data = input_data
        for layer in self.layers:
            output_data = layer(output_data)
        return output_data

    # Stochastic Gradient Descent used, as mentioned in the original algorithm
    @staticmethod
    def optimizer(learning_rate: float, momentum: float=0.0):
        return tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    # Loss_1(G) = mean(1-log(D(G(z_i)))) [Algorithm 1 in GAN Paper]
    # Loss_2(G) = -mean(log(D(G(z_i))))  [3. Adversarial Nets, Paragraph 3 in GAN Paper]
    @staticmethod
    def loss(generated_output):
        # generated_output = D(G(z_i)), output of D on the distribution of G
        # z_i sampled from Gaussian as part of mini batch
        loss_i = 1-tf.math.log(generated_output)
        loss = tf.math.reduce_mean(loss_i)
        return loss
