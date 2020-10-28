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
TRAIN_SIZE = 60000
BATCH_SIZE = 64
EPOCHS = 64
LATENT_DIMENSION = 2
N_EXAMPLES = 16
KL_WEIGHT = 0.1

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
from matplotlib import gridspec
import seaborn as sns
import imageio
import glob

sns.set_theme()

# %% Training constants
sample_tests = test_images[:N_EXAMPLES, :, :]
metrics_names = ['kl_loss', 'log_loss', 'loss']
batch_history = { name: [] for name in metrics_names }

# %% Fixed noise plot
plt.figure(figsize=(4,4))
for i in range(sample_tests.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(sample_tests[i] * 255.0, cmap='gray')
    plt.axis('off')

plt.savefig('analysis/test_samples.png')
plt.close()

def save_sample(decoder: Decoder, epoch: int, test_input: tf.Tensor, show: bool=False):
    predictions = decoder(test_input)

    plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i] * 255.0, cmap='gray')
        plt.axis('off')

    plt.savefig('data/image_at_epoch_{:04d}.png'.format(epoch))
    if show:
        plt.show()
    plt.close()

# %% Train
optimizer = tf.optimizers.SGD(learning_rate=0.002)
prior_distribution = encoder.prior()
save_sample(decoder, 0, encoder(sample_tests)[0], True)

# %%
for epoch in range(1, EPOCHS+1):
    print(f'\nepoch {epoch}/{EPOCHS}')
    progress_bar = keras.utils.Progbar(TRAIN_SIZE / BATCH_SIZE, stateful_metrics=metrics_names)

    for i, image_batch in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            posterior_sample, mean, var = encoder(image_batch)

            prior_sample = prior_distribution.sample()
            # latent_input_sample = posterior.sample()
            reconstructed_image_batch = decoder(posterior_sample)
            epsilon = 1e-8

            analytic_kl_divergence = -0.5 * -KL_WEIGHT * tf.reduce_sum(
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

            if tf.math.is_nan(loss):
                print(posterior_sample)
                print(mean)
                print(var)
                print(image_batch)
                print(reconstructed_image_batch)
                print(log_likelihood)
                print(kl_loss)
                print(log_loss)
                exit(0)

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

    save_sample(decoder, epoch, encoder(sample_tests)[0], epoch % 5 == 0)

print('')
if EPOCHS % 10 != 0:
    save_sample(decoder, EPOCHS, encoder(sample_tests)[0], True)

# %%
gs = gridspec.GridSpec(2, 2)
plt.figure(figsize=(20, 20))

ax = plt.subplot(gs[0, 0])
plt.title('KL Divergence')
plt.xlabel('Batch #')
plt.ylabel('Loss')
plt.plot(batch_history['kl_loss'])

ax = plt.subplot(gs[0, 1])
plt.title('Reconstruction Loss')
plt.xlabel('Batch #')
plt.ylabel('Loss')
plt.plot(batch_history['log_loss'], label='VAE Loss')

ax = plt.subplot(gs[1, :])
plt.title('Loss')
plt.xlabel('Batch #')
plt.ylabel('Loss')
plt.plot(batch_history['loss'], label='VAE Loss')

plt.savefig('analysis/loss.png')
plt.show()
plt.close()

# %%
with imageio.get_writer('analysis/vae.gif', mode='I') as writer:
    filenames = glob.glob('data/image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

# %%
digit_size = 28
grid_size = 20

norm = tfp.distributions.Normal(0, 1)
grid_x = norm.quantile(np.linspace(0.05, 0.95, grid_size))
grid_y = norm.quantile(np.linspace(0.05, 0.95, grid_size))
image_width = digit_size*grid_size
image_height = image_width
image = np.zeros((image_height, image_width))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z = np.array([[xi, yi]])
        x_decoded = decoder(z)
        digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
        image[i * digit_size: (i + 1) * digit_size,
              j * digit_size: (j + 1) * digit_size] = digit.numpy()

plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='Greys_r')
plt.axis('off')
plt.savefig('analysis/decoded_latent_space.png')
plt.show()
