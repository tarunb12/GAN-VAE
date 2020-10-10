# %% Import Libraries
import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

# %% Import D and E
from decoder import Decoder
from encoder import Encoder

# %% Constants
TRAIN_SIZE = 6000
BATCH_SIZE = 64
EPOCHS = 20
LATENT_DIMENSION = 2
N_EXAMPLES = 16
KL_WEIGHT = 3

# %% Load data, normalize and collapse all pixel values to 0 or 1
(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
train_images = train_images[:TRAIN_SIZE] / 255.
train_images = tf.round(train_images)
train_images = tf.dtypes.cast(train_images, tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
                               .shuffle(TRAIN_SIZE) \
                               .batch(BATCH_SIZE)
image_shape = train_images.shape[1:]

test_images = test_images[:N_EXAMPLES] / 255.
test_images = tf.round(test_images)
test_images = tf.dtypes.cast(test_images, tf.float32)

# %% Define E, D, and VAE
encoder = Encoder(image_shape, LATENT_DIMENSION)
decoder = Decoder(LATENT_DIMENSION, image_shape)

# %% Import plotting libraries
from matplotlib import pyplot as plt

# %% Training constants
sample_tests = test_images[:N_EXAMPLES, :, :]
metrics_names = ['kl_loss', 'log_loss', 'loss']
batch_history = { name: [] for name in metrics_names }

# %% Fixed noise plot
def view_sample(decoder: Decoder, epoch: int, test_input: tf.Tensor):
    predictions = decoder(test_input)

    plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i] * 255.0, cmap='gray')
        plt.axis('off')

    plt.savefig('data/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    plt.close()

    plt.figure(figsize=(4,4))
    for i in range(sample_tests.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(sample_tests[i] * 255.0, cmap='gray')
        plt.axis('off')

    plt.show()
    plt.close()


# %% Train
optimizer = tf.optimizers.Nadam()
prior_distribution = encoder.prior()
view_sample(decoder, 0, encoder(sample_tests)[0])

# %%
for epoch in range(1, EPOCHS+1):
    print(f'\nepoch {epoch}/{EPOCHS}')
    progress_bar = keras.utils.Progbar(TRAIN_SIZE / BATCH_SIZE, stateful_metrics=metrics_names)

    for i, image_batch in enumerate(train_dataset):
        if not image_batch: break
        with tf.GradientTape() as tape:
            posterior_sample, mean, var = encoder(image_batch)

            prior_sample = prior_distribution.sample()
            # latent_input_sample = posterior.sample()
            reconstructed_image_batch = decoder(posterior_sample)
            epsilon = 1e-8

            analytic_kl_divergence = 0.5 * KL_WEIGHT * tf.reduce_sum(
                1 + tf.math.log(var) - tf.math.square(mean) - var,
                axis=1,
            )
            log_likelihood = tf.reduce_sum(
                tf.math.add(
                    tf.math.multiply(image_batch, tf.math.log(reconstructed_image_batch + epsilon)),
                    tf.math.multiply((1-image_batch), tf.math.log(1-reconstructed_image_batch + epsilon)),
                ),
                axis=[1,2],
            )
            kl_loss = -tf.reduce_mean(analytic_kl_divergence)
            log_loss = -tf.reduce_mean(log_likelihood)
            loss = kl_loss + log_loss

            updated_metrics = {
                'kl_loss': kl_loss,
                'log_loss': log_loss,
                'loss': loss,
            }

            # Record loss history
            for metric in batch_history:
                batch_history[metric].append(updated_metrics[metric])

            metric_values = updated_metrics.items()
            progress_bar.update(i, values=metric_values)

        model_vars = [*encoder.variables, *decoder.variables]
        grad = tape.gradient(loss, model_vars)
        optimizer.apply_gradients(zip(grad, model_vars))

    view_sample(decoder, epoch, encoder(sample_tests)[0])

if EPOCHS % 10 != 0:
    view_sample(decoder, EPOCHS, encoder(sample_tests)[0])

# %%
plt.figure(figsize=(20, 20))
plt.plot(batch_history['loss'], label='VAE Loss')
plt.xlabel('Batch #')
plt.ylabel('Loss')
plt.legend()
plt.savefig('data/loss.png', transparent=True)
plt.show()
plt.close()
# %%
view_sample(decoder, 0, tf.random.normal([N_EXAMPLES, LATENT_DIMENSION]))

# %%
