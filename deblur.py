from imageio import imread
import sol5_utils
import random
from skimage.color import rgb2gray
import scipy.ndimage.filters
import numpy as np

from keras import Model
from keras.layers import Conv2D, Activation, Input, Add
from keras.optimizers import Adam

TRANING_SET_PERCENT = 0.8
VALIDATION_PERCENT = 1 - TRANING_SET_PERCENT

####################################################################################################################

def read_image(filename, representation):
    im = imread(filename)
    if (im.dtype == np.uint8 or im.dtype == int or im.np.matrix.max > 1):
        im = im.astype(np.float64) / 255

    if ((representation == 1) and (len(im.shape) >= 3)):  # process image into gray scale

        im = rgb2gray(im)

    return im


####################################################################################################################

### 3-DATASET HANDLING : ###


def random_coor1(im, crop_size):

    random_x = np.random.randint(im.shape[0] - (3 * crop_size[0]))
    random_y = np.random.randint(im.shape[1] - (3 * crop_size[1]))
    return random_x, random_y

def random_coor2(crop_size):

    random_x = np.random.randint(crop_size[0] * 2)
    random_y = np.random.randint(crop_size[1] * 2)
    return random_x, random_y


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    loaded_images = dict()
    while True:
        index = 0

        source_batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
        target_batch = np.zeros((batch_size, crop_size[0], crop_size[1], 1))

        # randomly choosing images for batch:
        random_images_keys = np.random.choice(filenames, size=batch_size)
        for image_path in random_images_keys:
            patch_size = (crop_size[0]*3, crop_size[1]*3)

            # making sure we dont read image twice:
            if image_path not in loaded_images:
                image = read_image(image_path, 1)
                loaded_images[image_path] = image
            else:
                image = loaded_images[image_path]

            #choose random coordinates for patches :
            rand_x1 , rand_y1 = random_coor1(image,crop_size)
            rand_x2,rand_y2 = random_coor2(crop_size)


            # create the patch from original:
            total_x = rand_x2 + rand_x1
            total_y = rand_y2 + rand_y1
            patch = image[total_x: total_x + crop_size[0], total_y: total_y + crop_size[1]]
            patch = patch.reshape(crop_size[0],  crop_size[1], 1)- 0.5
            target_batch[index] += patch


            # create corrupted :
            patch_for_corrupt = image[rand_x1: rand_x1 + patch_size[0], rand_y1: rand_y1 + patch_size[1]]
            patch_for_corrupt = corruption_func(patch_for_corrupt)
            corrupted = patch_for_corrupt[rand_x2: rand_x2 + crop_size[0], rand_y2: rand_y2 + crop_size[1]]
            corrupted = corrupted.reshape(crop_size[0],  crop_size[1], 1)
            source_batch[index] += corrupted- 0.5

            index = index+1

        yield (source_batch, target_batch)



####################################################################################################################
### 4-Neural Network Model: ###


def resblock(input_tensor, num_channels):
    X = input_tensor
    O = after_block_manipulation(input_tensor, num_channels)
    ans = Add()([X, O])
    return Activation('relu')(ans)


def after_block_manipulation(input_tensor, num_channels):
    ans = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    ans = Activation('relu')(ans)
    ans = Conv2D(num_channels, (3, 3), padding='same')(ans)
    return ans


def build_nn_model(height, width, num_channels, num_res_blocks):
    input_tensor = Input(shape=(height, width, 1))

    # making first layers before the res_blocks:
    layers = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    layers = Activation('relu')(layers)

    # adding the res_blocks to the network :
    for _ in range(num_res_blocks):
        layers = resblock(layers, num_channels)

    # making the final output with one channel:
    layers = Conv2D(1, (3, 3), padding='same')(layers)
    layers = Add()([input_tensor, layers])

    return Model(inputs=input_tensor, outputs=layers)


####################################################################################################################
### 5-Training Networks for Image Restoration : ###

def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):

    # # spliting to traning and validation set  after shuffeling the data to get better distribution:
    random.shuffle(images)
    slice_index = int((len(images) * TRANING_SET_PERCENT))
    training_set = images[0:slice_index]
    validation_set = images[slice_index:len(images)]

    #loading the data sets:
    load_training = load_dataset(training_set, batch_size, corruption_func,(model.input_shape[1], model.input_shape[2]))
    load_validation = load_dataset(validation_set, batch_size, corruption_func,(model.input_shape[1],
                                                                                model.input_shape[2]))

    # doing the training:
    model.compile(Adam(beta_2=0.9), 'mean_squared_error')
    model.fit_generator(generator=load_training, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=load_validation, validation_steps=num_valid_samples)


### 6-Image Restoration of Complete Images : ###

def restore_image(corrupted_image, base_model):
    # first we adjust the model to the new size :
    a = Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)

    result_im = new_model.predict(
        (corrupted_image).reshape(tuple(list(tuple([1] + list(corrupted_image.shape))) + [1])) - 0.5)[0, :, :, 0]
    return np.clip((result_im + 0.5), 0, 1).astype(np.float64)


### 7-Application to Image Denoising and Deblurring : ###

def add_gaussian_noise(image, min_sigma, max_sigma):
    # randomly sample a value of sigma, uniformly distributed between min_sigma and max_sigma :
    sigma = np.random.uniform(min_sigma, max_sigma)

    # adding to every pixel of the input image a zero-mean gaussian random variable with
    # standard deviation equal to sigma :

    random_add = np.random.normal(0, sigma, image.shape)
    result = image + random_add

    # Before returning the results, the values should be rounded to the nearest fraction i and clipped to [0,1] :
    result = np.round(result * 255) / 255
    return np.clip(result, 0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    images = sol5_utils.images_for_denoising()
    corruption_func = lambda image: add_gaussian_noise(image, 0.0, 0.2)
    model = build_nn_model(24, 24, 48, num_res_blocks)
    batch_size = 100
    steps_per_epoch = 100
    num_epochs = 5
    num_valid_samples = 1000
    if quick_mode:
        batch_size = 10
        steps_per_epoch = 3
        num_epochs = 2
        num_valid_samples = 30

    train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    # create filter:
    filter = sol5_utils.motion_blur_kernel(kernel_size, angle)

    # do the bluring:
    return scipy.ndimage.filters.convolve(image, filter)


def random_motion_blur(image, list_of_kernel_sizes):
    # choosing random parameters:
    angle = np.random.uniform(0, np.pi)
    kernel_size = random.choice(list_of_kernel_sizes)

    result = add_motion_blur(image, kernel_size, angle)
    result = np.round(result * 255) / 255
    return np.clip(result, 0, 1)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    images = sol5_utils.images_for_deblurring()
    corruption_func = lambda image: random_motion_blur(image, [7])
    model = build_nn_model(16, 16, 32, num_res_blocks)
    batch_size = 100
    steps_per_epoch = 100
    num_epochs = 10
    num_valid_samples = 1000
    if quick_mode:
        batch_size = 10
        steps_per_epoch = 3
        num_epochs = 2
        num_valid_samples = 30

    train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples)
    return model




# def check_network_denoising():
#
#     model_noise = learn_denoising_model(quick_mode=False)
#     im1_corrupt = read_image("examples/163004_2_corrupted_0.10.png", 1)
#     im1_fixed = read_image("examples/163004_3_fixed_0.10_5.png", 1)
#     im1_out = restore_image(im1_corrupt, model_noise)
#     plt.imshow(np.hstack((im1_corrupt, im1_out, im1_fixed)), cmap="gray")
#     plt.axis('off')
#     plt.show()
#
#
# def check_network_deblurring():
#     model_blur = learn_deblurring_model(quick_mode=False)
#     im2_corrupt = read_image("examples/0000018_2_corrupted.png", 1)
#     im2_fixed = read_image("examples/0000018_3_fixed.png", 1)
#     im2_out = restore_image(im2_corrupt, model_blur)
#     plt.imshow(np.hstack((im2_corrupt, im2_out, im2_fixed)), cmap="gray")
#     plt.axis('off')
#     plt.show()
#
# if __name__ == "__main__" :
#
#     # check_network_deblurring()
#     check_network_denoising()

    # validation_error_denoise = []
    # validation_error_deblur = []
    # for i in range(1, 6):
    #     denoise_model = learn_denoising_model(i)
    #     deblur_model = learn_deblurring_model(i)
    #     validation_error_denoise.append(denoise_model.history.history['val_loss'][-1])
    #     validation_error_deblur.append(deblur_model.history.history['val_loss'][-1])
    #
    # arr = np.arange(1, 6)
    #
    # plt.plot(arr, validation_error_denoise)
    # plt.title('validation error - denoise')
    # plt.xlabel('number res blocks')
    # plt.ylabel('validation loss denoise')
    # plt.savefig('denoise.png')
    # plt.show()
    #
    # plt.plot(arr, validation_error_deblur)
    # plt.title('validation error - deblur')
    # plt.xlabel('number res blocks')
    # plt.ylabel('validation loss deblur')
    # plt.savefig('deblur.png')
    # plt.show()