# %% Import libraries
import tensorflow as tf
from tensorflow import keras

# %% Discriminator model D
class Discriminator(tf.Module):
    def __init__(self, input_shape: tuple) -> None:
        super().__init__()
        # Network layers
        self.flatten1 = keras.layers.Flatten(input_shape=input_shape)
        self.dense1 = keras.layers.Dense(units=512)
        self.dense2 = keras.layers.Dense(units=256)
        self.output_data = keras.layers.Dense(units=1)
        self.dropout = tf.keras.layers.Dropout(0.3)

    @tf.Module.with_name_scope
    def __call__(self, input_data, training=False):
        # Activations
        l_relu = keras.layers.ELU()
        sigmoid = keras.layers.Activation(tf.nn.sigmoid)

        # Feed forward
        out = input_data
        out = self.flatten1(out)
        out = self.dense1(out)
        out = l_relu(out)
        out = self.dropout(out, training=training)
        out = self.dense2(out)
        out = l_relu(out)
        out = self.dropout(out, training=training)
        out = self.output_data(out)
        out = sigmoid(out)

        return out

    # Stochastic Gradient Descent used, as mentioned in the original algorithm
    @staticmethod
    def optimizer(learning_rate: float, momentum: float=0.0):
        # return tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        return tf.optimizers.Adam(.0001)

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
