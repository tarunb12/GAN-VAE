# %% Import Libraries
import os
import tensorflow as tf
from tensorflow import keras

# %% Import D and G
from generate import Generator
from discriminate import Discriminator

# %% Constants
NOISE_DIMENSION = 100
BUFFER_SIZE = 10000
BATCH_SIZE = 256
EPOCHS = 20
N_EXAMPLES = 12
G_LEARNING_RATE = 0.01
D_LEARNING_RATE = 0.01

# %% Load data, normalize to [0, 1], and shuffle/split dataset
(train_images, _), (test_images, _) = keras.datasets.mnist.load_data()
train_images = tf.cast(train_images[:BUFFER_SIZE] / 255., tf.float32)
num_images = train_images.shape[0]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images) \
                               .shuffle(BUFFER_SIZE) \
                               .batch(BATCH_SIZE)
image_shape = train_images.shape[1:]

# %% Define G
generator = Generator((NOISE_DIMENSION,), image_shape)
g_optimizer = generator.optimizer(G_LEARNING_RATE, 0.5)

# %% Define D
discriminator = Discriminator(image_shape)
d_optimizer = discriminator.optimizer(D_LEARNING_RATE, 0.5)

# %% Set training checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    generator_optimizer=g_optimizer,
    discriminator_optimizer=d_optimizer
)

# %% Plotting libraries
from matplotlib import pyplot as plt

# %% Training constants
fixed_noise = tf.random.normal([N_EXAMPLES, NOISE_DIMENSION])
metrics_names = ['g_loss', 'd_loss', 'acc', 'real_acc', 'fake_acc']
batch_history = { name: [] for name in metrics_names }
epoch_history = { name: [] for name in metrics_names }

# %% Fixed noise plot
def view_sample(generator: Generator, epoch: int, test_input: tf.Tensor):
    predictions = generator(test_input, training=False)

    plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i] * 255.0, cmap='gray')
        plt.axis('off')

    plt.savefig('pics/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    plt.close()

# %% Train model
view_sample(generator, 1, fixed_noise)

for epoch in range(1, EPOCHS+1):
    print(f'\nepoch {epoch}/{EPOCHS}')
    progress_bar = keras.utils.Progbar(num_images / BATCH_SIZE, stateful_metrics=metrics_names)

    for i, image_batch in enumerate(train_dataset):

        num_samples = image_batch.shape[0]
        noise = tf.random.normal([num_samples, NOISE_DIMENSION])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            # Train G on noise
            generated_images = generator(noise, training=True)
            # Train D on training images
            trained_output = discriminator(image_batch, training=True)
            # Train D on generated images
            generated_output = discriminator(generated_images, training=True)

            # Calculate loss
            g_loss = generator.loss(generated_output)
            d_loss = discriminator.loss(trained_output, generated_output)

            # Trained images fed into D have output 1, images from G's noisy
            # distribution should have output 0 from D
            correct_trained_output = tf.ones_like(trained_output)
            correct_generated_ouput = tf.zeros_like(generated_output)

            real_acc = tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.round(trained_output), correct_trained_output), tf.float32))
            fake_acc = tf.math.reduce_mean(tf.cast(tf.math.equal(tf.math.round(generated_output), correct_generated_ouput), tf.float32))
            acc = 0.5 * (real_acc + fake_acc)

            updated_metrics = {
                'g_loss': g_loss,
                'd_loss': d_loss,
                'acc': acc,
                'real_acc': real_acc,
                'fake_acc': fake_acc,
            }

            # Record loss history
            for metric in batch_history:
                batch_history[metric].append(updated_metrics[metric])

            metric_values = updated_metrics.items()
            progress_bar.update(i, values=metric_values)        

        # https://www.tensorflow.org/api_docs/python/tf/GradientTape#gradient
        grad_g = g_tape.gradient(g_loss, generator.variables)
        grad_d = d_tape.gradient(d_loss, discriminator.variables)

        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#apply_gradients
        g_optimizer.apply_gradients(zip(grad_g, generator.variables))
        d_optimizer.apply_gradients(zip(grad_d, discriminator.variables))    

    if epoch % 10 == 0:
        view_sample(generator, epoch, fixed_noise)

# %% Loss plots
plt.figure(figsize=(10, 8))
plt.plot(batch_history['g_loss'], label='Generator Loss')
plt.plot(batch_history['d_loss'], label='Discriminator Loss')
plt.xlabel('Batch #')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close()

# %% Accuracy plots for D
plt.figure(figsize=(10, 8))
plt.plot(batch_history['acc'], label='Total Accuracy')
plt.plot(batch_history['real_acc'], label='Training Data Accuracy')
plt.plot(batch_history['fake_acc'], label='Generated Data Accuracy')
plt.xlabel('Batch #')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()

# %%
x = tf.random.normal([50, 28, 28])
discriminator(x, training=False)
# %%
plt.figure(figsize=(4,4))
plt.subplot(4, 4, 1)
plt.imshow(x[5] * 255.0, cmap='gray')
plt.axis('off')
plt.show()
