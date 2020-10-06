# %% Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras


# %% Generator model G
class Generator(tf.Module):
    def __init__(self, noise_shape: tuple, output_shape: tuple) -> None:
        super().__init__()
        # Network layers
        self.dense1 = keras.layers.Dense(units=128, input_shape=noise_shape)
        self.dense2 = keras.layers.Dense(units=256)
        self.dense3 = keras.layers.Dense(units=512)
        self.dense4 = keras.layers.Dense(units=np.prod(output_shape))
        self.output_data = keras.layers.Reshape(target_shape=output_shape)

    @tf.Module.with_name_scope
    def __call__(self, input_data, training=False) -> tf.Tensor:
        batch_normalization = keras.layers.BatchNormalization()
        # Activations
        relu = keras.layers.ReLU()
        sigmoid = keras.layers.Activation(tf.nn.sigmoid)

        # Feed forward
        out = input_data
        out = self.dense1(out)
        out = batch_normalization(out)
        out = relu(out)
        out = self.dense2(out)
        out = relu(out)
        out = self.dense3(out)
        out = relu(out)
        out = self.dense4(out)
        out = relu(out)
        out = self.output_data(out)
        out = sigmoid(out)

        return out

    # Stochastic Gradient Descent used, as mentioned in the original algorithm
    @staticmethod
    def optimizer(learning_rate: float, momentum: float=0.0):
        # return tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        return tf.optimizers.Adam(.0001)

    # Loss_1(G) = mean(1-log(D(G(z_i)))) [Algorithm 1 in GAN Paper]
    # Loss_2(G) = -mean(log(D(G(z_i))))  [3. Adversarial Nets, Paragraph 3 in GAN Paper]
    @staticmethod
    def loss(generated_output):
        # generated_output = D(G(z_i)), output of D on the distribution of G
        # z_i sampled from Gaussian as part of mini batch
        loss_i = 1-tf.math.log(generated_output)
        loss = tf.math.reduce_mean(loss_i)
        return loss

# %%
