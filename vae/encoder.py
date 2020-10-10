# %% Import libraries
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

# %%
class Encoder(tf.Module):
    def __init__(self, input_shape: tuple, latent_dimension: int) -> None:
        super().__init__()
        self.latent_dimension = latent_dimension
        self.layers = [
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(units=256),
            keras.layers.Activation(tf.nn.tanh),
            keras.layers.Dense(units=latent_dimension*2),
            keras.layers.Reshape(target_shape=(latent_dimension, 2))
        ]

    # Posterior: z ~ = N(mean, var)
    @tf.Module.with_name_scope
    def __call__(self, input_data: tf.Tensor) -> tfp.distributions.Distribution:
        output_data = input_data
        for layer in self.layers:
            output_data = layer(output_data)
        mean, log_var = output_data[:, :, 0], output_data[:, :, 1]
        epsilon = tf.random.normal(shape=log_var.shape)
        # sd = e^(log(var^.5)) = e^(0.5*log(var))
        stddev = tf.math.multiply(epsilon, tf.math.exp(0.5 * log_var))
        # Reparameterization: z ~ N(mean, stddev * epsilon)
        return mean + stddev * epsilon, mean, tf.math.square(stddev)

    # Prior: z ~ N(0, I)
    def prior(self) -> tfp.distributions.Distribution:
        mean = tf.zeros(self.latent_dimension)
        stddev = tf.ones(self.latent_dimension)
        return tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=stddev)
