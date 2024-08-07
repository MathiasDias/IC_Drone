import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
from datetime import datetime
# NAO UTILIZAR GPUS
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# datetime object containing current date and time
now = datetime.now()
dt_str = now.strftime("%d-%m-%Y-%H-%M-%S")
dir_res = 'resultados/res'+'_'+ dt_str
os.mkdir(dir_res)

IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_CLASSES = 4
DATA_DIR = "instance-level_human_parsing/instance-level_human_parsing"
print("DATADIR: ", DATA_DIR)
NUM_TRAIN_IMAGES = 31
NUM_VAL_IMAGES = 14

train_images = sorted(glob(os.path.join(DATA_DIR, "Training/Images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "Training/Category_ids/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "Validation/Images/*")))
val_masks = sorted(glob(os.path.join(DATA_DIR, "Validation/Category_ids/*")))

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127.5 - 1
    return image

def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def FCN(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    x = convolution_block(model_input, num_filters=256, kernel_size=7)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = convolution_block(x, num_filters=256, kernel_size=5)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = convolution_block(x, num_filters=256, kernel_size=3)

    # Adjust the upsampling operation
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation='softmax')(x)

    return keras.Model(inputs=model_input, outputs=x)

print("IMAGE_SIZE: ", IMAGE_SIZE)

model = FCN(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)

loss = keras.losses.CategoricalCrossentropy()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
    run_eagerly=True
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=500)
model.save(os.path.join(dir_res, 'model'))

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.savefig(os.path.join(dir_res, 'training_loss_batch_' + str(BATCH_SIZE) + '.png'))
plt.close()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.savefig(os.path.join(dir_res, 'training_acc_batch_' + str(BATCH_SIZE) + '.png'))
plt.close()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.savefig(os.path.join(dir_res, 'validation_loss_batch_' + str(BATCH_SIZE) + '.png'))
plt.close()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.savefig(os.path.join(dir_res, 'validation_acc_batch_' + str(BATCH_SIZE) + '.png'))
plt.close()

# Loading the Colormap
colormap = loadmat(
    "./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat"
)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def plot_samples_matplotlib(display_list, file_name, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])

    plt.savefig(file_name)


def plot_predictions(images_list, colormap, model):

    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 4)
        overlay = get_overlay(image_tensor, prediction_colormap)
        file_name = os.path.join(dir_res,os.path.basename(image_file))
        print("file_name:",file_name)
        plot_samples_matplotlib(
            [image_tensor, overlay, prediction_colormap], file_name, figsize=(18, 14)
        )

#plot_predictions(train_images[:5], colormap, model=model)
plot_predictions(val_images, colormap, model=model)
