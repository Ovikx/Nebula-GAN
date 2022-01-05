import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras import backend
import numpy as np
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 4
SAVE_INTERVAL = 10
MODEL_SAVE_INTERVAL = 50
IMAGE_SIZE = (256, 256)
IMAGE_SHAPE = IMAGE_SIZE + (3,)

# Load the dataset from directory and normalize it
train_images = image_dataset_from_directory(
    directory='nebula_images',
    color_mode='rgb',
    labels=None,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
).map(lambda data : data / 255.0)

# The target shape of the generated image is (64, 64, 3)
# (8, 8, 512) -> (8, 8, 256) -> (16, 16, 128) -> (32, 32, 64) -> (64, 64, 3)
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*2048, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # This reshape layer is crucial for creating the basis of the image
    model.add(layers.Reshape(target_shape=(8, 8, 2048)))
    # Make sure that the image is the correct shape
    assert model.output_shape == (None, 8, 8, 2048)

    model.add(layers.Conv2DTranspose(
        filters=1024,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        use_bias=False
    ))
    assert model.output_shape == (None, 8, 8, 1024)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        filters=512,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=False
    ))
    assert model.output_shape == (None, 16, 16, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        filters=256,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=False
    ))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        filters=128,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=False
    ))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=False
    ))
    assert model.output_shape == (None, 128, 128, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        filters=3,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='same',
        use_bias=False,
        activation='sigmoid'
    ))
    assert model.output_shape == (None, 256, 256, 3)
    return model

def build_discriminator():
    # If the discriminator's loss is too low for the generator's improvement, increase the standard deviation of the Gaussian noise
    STDEV = 0
    model = tf.keras.Sequential([
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same',
            input_shape=IMAGE_SHAPE
        ),
        layers.GaussianNoise(STDEV),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same'
        ),
        layers.GaussianNoise(STDEV),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same'
        ),
        layers.GaussianNoise(STDEV),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])

    return model

generator = build_generator()
discriminator = build_discriminator()

loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_function(tf.fill(real_output.shape, 0.8), real_output)
    fake_loss = loss_function(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss

def generator_loss(fake_output):
    return loss_function(tf.ones_like(fake_output), fake_output)

gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)

seed = tf.random.normal([16, 100])

checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    gen_opt=gen_opt,
    disc_opt=disc_opt
)
manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory='./training_checkpoints', max_to_keep=3)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_guess = discriminator(images, training=True)
        fake_guess = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_guess)
        disc_loss = discriminator_loss(real_guess, fake_guess)
    
    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

    return (gen_loss, disc_loss)

def save_predictions(epoch, z):
    predictions = (generator(z, training=False).numpy()*255).astype('int32')
    fig = plt.figure(figsize=(12, 12))

    for i, image in enumerate(predictions):
        plt.subplot(4, 4, i+1)
        plt.imshow(image, cmap=None)
        plt.axis('off')
    
    plt.savefig(f'seq2/gen_{epoch}')
    plt.close()

def train(ds, epochs):
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f'Restored models and optimizers from {manager.latest_checkpoint}')
    else:
        print('Initializing fresh models and optimizers')

    for epoch in range(epochs):
        for image_batch in ds:
            losses = train_step(image_batch)
        
        if epoch % SAVE_INTERVAL == 0:
            save_predictions(epoch, seed)
        
        if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            if (epoch + 1) >= 700:
                generator.save(f'models2/generators/nebula_generator_{epoch+1}.h5')
            save_path = manager.save()
            print(f'Saved checkpoint to {save_path}')
        
        print(f'Epoch {epoch} : Gen loss = {losses[0]} || Disc loss = {losses[1]}')

train(train_images, 2000)