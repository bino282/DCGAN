from PIL import Image
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import *
from keras.models import Model
import numpy as np
import time
import glob
from model import *
from collections import deque
import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def norm_img(img):
    img = (img / 127.5) - 1
    return img

def generate_images(generator, save_dir):
    noise = gen_noise(batch_size,noise_shape)
    #using noise produced by np.random.uniform - the generator seems to produce same image for ANY noise - 
    #but those images (even though they are the same) are very close to the actual image - experiment with it later.
    fake_data_X = generator.predict(noise)
    print("Displaying generated images")
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0],16,replace=False)
    for i in range(16):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir+str(time.time())+"_GENERATEDimage.png",bbox_inches='tight',pad_inches=0)
    plt.show()
def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 
def gen_noise(batch_size, noise_shape):
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)
def sample_from_dataset(batch_size, image_shape, data_dir=None, data = None):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist,batch_size)
    for index,img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB')
        image = np.asarray(image)
        image = norm_img(image)
        sample[index,...] = image
    return sample
noise_shape = (1,1,100)
num_steps = 10000
batch_size = 128
img_save_dir = '../local/img_save_dir/'
model_saved_dir = '../local/model_saved/'
log_dir = '../local/logs/'
image_shape = (64,64,3)
data_dir =  "../local/gan/data_dir/*.png"
discriminator = discriminator_model(image_shape)
generator = generator_model(noise_shape)

discriminator.trainable = False
opt = Adam(lr=0.00015, beta_1=0.5)
gen_inp = Input(shape=noise_shape)
GAN_inp = generator(gen_inp)
GAN_opt = discriminator(GAN_inp)
gan = Model(input = gen_inp, output = GAN_opt)
gan.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
gan.summary()
avg_disc_fake_loss = deque([0], maxlen=250)     
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)
for step in range(num_steps):
    tot_step = step
    print("Begin step: ", tot_step)
    step_begin_time = time.time()
    real_data_X = sample_from_dataset(batch_size, image_shape, data_dir = data_dir)
    noise = gen_noise(batch_size,noise_shape)
    fake_data_X = generator.predict(noise)
    data_X = np.concatenate([real_data_X,fake_data_X])
    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
    fake_data_Y = np.random.random_sample(batch_size)*0.2
    data_Y = np.concatenate((real_data_Y,fake_data_Y))
    discriminator.trainable = True
    generator.trainable = False
    dis_metrics_real = discriminator.train_on_batch(real_data_X,real_data_Y)   #training seperately on real
    dis_metrics_fake = discriminator.train_on_batch(fake_data_X,fake_data_Y)   #training seperately on fake
    print("Disc: real loss: %f fake loss: %f" % (dis_metrics_real[0], dis_metrics_fake[0]))
    avg_disc_fake_loss.append(dis_metrics_fake[0])
    avg_disc_real_loss.append(dis_metrics_real[0])

    generator.trainable = True
    GAN_X = gen_noise(batch_size,noise_shape)
    GAN_Y = real_data_Y
    discriminator.trainable = False
    gan_metrics = gan.train_on_batch(GAN_X,GAN_Y)
    print("GAN loss: %f" % (gan_metrics[0]))
    text_file = open(log_dir+"training_log.txt", "a")
    text_file.write("Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (tot_step, dis_metrics_real[0], dis_metrics_fake[0],gan_metrics[0]))
    text_file.close()
    avg_GAN_loss.append(gan_metrics[0])
    end_time = time.time()
    diff_time = int(end_time - step_begin_time)
    print("Step %d completed. Time took: %s secs." % (tot_step, diff_time))

generator.save(model_saved_dir+'generator_model.h5')
discriminator.save(model_saved_dir+'discriminator_model.h5')
#generate final sample images
for i in range(10):
    generate_images(generator, img_save_dir)

#Generating GIF from PNG
images = []
all_data_dirlist = list(glob.glob(img_save_dir+"*_image.png"))
for filename in all_data_dirlist:
    img_num = filename.split('\\')[-1][0:-10]
    if (int(img_num) % 100) == 0:
        images.append(imageio.imread(filename))
imageio.mimsave(img_save_dir+'movie.gif', images) 