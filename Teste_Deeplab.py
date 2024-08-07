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
#tf.config.run_functions_eagerly(True)
#tf.data.experimental.enable_debug_mode()

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

#print("GLOB:", os.path.join(DATA_DIR, "Training/Images/*"))

train_images = sorted(glob(os.path.join(DATA_DIR, "Training/Images/*")))[:NUM_TRAIN_IMAGES]
#print("train_images: ", train_images)
train_masks = sorted(glob(os.path.join(DATA_DIR, "Training/Category_ids/*")))[:NUM_TRAIN_IMAGES]
#NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
val_images = sorted(glob(os.path.join(DATA_DIR, "Validation/Images/*")))
val_masks = sorted(glob(os.path.join(DATA_DIR, "Validation/Category_ids/*")))
#print("val_images: ", val_images)

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    #print("image: ", image)

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
    #print("image_list: ", image_list)
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    #print("image_list:", image_list)
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


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

print("IMAGE_SIZE: ",  IMAGE_SIZE)

model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
#model.summary()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
    run_eagerly=True
)

##Criar o Load e o Save para deixar salvo os pesos e carregar as imagens mais rapido depois, os links estão no email dia 9/06

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=30)

history = model.fit(train_dataset, validation_data=val_dataset, callbacks=[callback], epochs=300)
model.save(os.path.join(dir_res,'model'))

# Plotar Training and Validation Loss no mesmo grafico
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.ylim(0, 1)
plt.legend()
plt.savefig(os.path.join(dir_res, 'loss_batch_' + str(BATCH_SIZE) + '.png'))
plt.close()

# Plotar Training and Validation Accuracy no mesmo grafico
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(dir_res, 'accuracy_batch_' + str(BATCH_SIZE) + '.png'))
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

def plot_images(images_list, colormap, model):

    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 4)
        file_name = os.path.join(dir_res,os.path.basename(image_file))
        print("file_name:",file_name)
        plot_samples_matplotlib(
            prediction_colormap, file_name, figsize=(18, 14)
        )

#plot_predictions(train_images[:5], colormap, model=model)
plot_predictions(val_images, colormap, model=model)
plot_images(val_images, colormap, model=model)
