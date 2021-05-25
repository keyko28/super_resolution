from generator import Generator
from discriminator import Discriminator
import config as cfg
from build_loss import Loss
from utils import load_train_images_names, get_data, stdout_write, generator_predict
from build_model import GanModel
from tqdm import tqdm
import numpy as np


def train(epochs: int, str, paths: list, model_save_dir: str, batch_size: int = 64):

    # paths: lr_train, hr_train, lr_test, hr_test, extension
    images: list = load_train_images_names(*paths)
    x_train_lr, x_train_hr, x_test_lr, x_test_hr  = images
    max_size = max([len(image_set) for image_set in images])

    batch_count =  max_size // batch_size

    loss = Loss()

    generator = Generator(mean_df = cfg.MEAN, std_df = cfg.STD)
    generator._build_gen()

    discriminator = Discriminator()
    discriminator._build_discriminator()

    gan = GanModel()
    gan._build_gan(generator = generator, discriminator = discriminator, loss=loss)


    loss_file = open(model_save_dir + 'losses_info.txt', 'w+')
    loss_file.close()

    #losses
    gan_loss = None
    discriminator_loss = None

    for epoch in range(epochs):

        stdout_write(['~'*20, f'Epoch: {epoch}', '~'*20, '\n'])

        for _ in tqdm(range(batch_count)):

            random_images = np.random.randint(0, max_size, size = batch_size)

            batch_hr = [x_train_hr[i] for i in random_images]
            batch_lr = [x_train_lr[i] for i in random_images]

            batch_hr = get_data(batch_hr)
            batch_lr = get_data(batch_lr)

            generated_image_sr = generator_predict(generator.model, batch_lr)

            real_data_y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_data_y = np.random.random_sample(batch_size) * 0.2

            discriminator.model.trainable = True

            d_loss_real = discriminator.model.train_on_batch(list(batch_hr), real_data_y)
            d_loss_fake = discriminator.model.train_on_batch(generated_image_sr, fake_data_y)

            discriminator_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            random_images = np.random.randint(0, max_size, size = batch_size)

            batch_hr = [x_train_hr[i] for i in random_images]
            batch_lr = [x_train_lr[i] for i in random_images]

            batch_hr = get_data(batch_hr)
            batch_lr = get_data(batch_lr)

            gan_y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2

            discriminator.model.trainable = False

            gan_loss = gan.model.train_on_batch(list(batch_lr), [list(batch_hr), gan_y])

        stdout_write([f'discriminator loss: {discriminator_loss}'])
        stdout_write([f'gan loss: {gan_loss}'])

        with open (model_save_dir + 'losses.txt', 'a') as f:
            f.write(f'epoch: {epoch}: gan_loss: {gan_loss}; discriminator loss: {discriminator_loss}')

        if epoch % 5 == 0:
            generator.model.save(model_save_dir + f'gen_model_{epoch}.h5')
            discriminator.model.save(model_save_dir + f'discr_model_{epoch}.h5')


    


