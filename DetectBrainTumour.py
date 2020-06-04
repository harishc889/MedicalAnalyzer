import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, \
    Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.utils import shuffle
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from os import listdir

from Gui_graphplot import createWindowforGraph1


def crop_brain_contour(image, plot):
    # import imutils
    # import cv2
    # from matplotlib import pyplot as plt

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')

        plt.show()

    return new_image


def load_data(dir_list, image_size):
    # load all images in a directory
    X = []
    y = []
    image_width, image_height = image_size

    for directory in dir_list:
        for filename in listdir(directory):
            # load the image
            # if(filename!='Thumbs.db'):
            #print(filename)
            image = cv2.imread(os.path.join(directory, filename))

            # crop the brain and ignore the unnecessary rest part of the image

            image = crop_brain_contour(image, False)
            # resize image

            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])

    X = np.array(X)
    y = np.array(y)

    # Shuffle the data
    X, y = shuffle(X, y)

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')

    return X, y


def plot_sample_images(X, y, n=50):
       for label in [0, 1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]

        columns_n = 10
        rows_n = int(n / columns_n)

        plt.figure(figsize=(20, 10))

        i = 1  # current plot
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])

            # remove ticks
            plt.tick_params(axis='both', which='both',
                            top=False, bottom=False, left=False, right=False,
                            labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            i += 1

        label_to_str = lambda label: "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()



def prepare_data():
    augmented_path = "augmented data//"
    # augmented data (yes and no) contains both the original and the new generated examples
    augmented_yes = augmented_path + 'yes'
    augmented_no = augmented_path + 'no'

    IMG_WIDTH, IMG_HEIGHT = (240, 240)

    X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))
    #plot_sample_images(X, y, 50)
    return X,y


def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    print(f'xtrain is: {X_train.shape}')

    return X_train, y_train, X_val, y_val, X_test, y_test


#X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, 0.2)


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s, 1)}"


def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)

    score = f1_score(y_true, y_pred)

    return score


def build_model(input_shape):

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)  # shape=(?, 240, 240, 3)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)  # shape=(?, 244, 244, 3)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)  # shape=(?, 238, 238, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool0')(X)  # shape=(?, 59, 59, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4), name='max_pool1')(X)  # shape=(?, 14, 14, 32)

    # FLATTEN X
    X = Flatten()(X)  # shape=(?, 6272)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid', name='fc')(X)  # shape=(?, 1)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')

    return model

def training_model(model, X_train, y_train, X_val,y_val):
    start_time = time.time()
    model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")
    return model

#IMG_SHAPE = (240, 240, 3)
#model = build_model(IMG_SHAPE)
#model.summary()

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# tensorboard
log_file_name = 'fbrain_tumor_detection_cnn_{int(time.time())}'
tensorboard = TensorBoard(log_dir=f'logs\{log_file_name}')

# checkpoint
# unique file name that will include the epoch and the validation (development) accuracy
filepath = "cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}\\"
# save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint(
    "models\{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))



#history = model.history.history

#for key in history.keys():
#    print(key)


def plot_metrics(history):
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']

    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

def loading_best_model(X_test,y_test,X_val,y_val,history):
    plot_metrics(history)
    best_model = load_model(filepath='saved_model/cnn-parameters-improvement-23-0.91.model')
    best_model.metrics_names
    loss, acc = best_model.evaluate(x=X_test, y=y_test)
    print(f"Test Loss = {loss}")
    print(f"Test Accuracy = {acc}")
    y_test_prob = best_model.predict(X_test)
    f1score = compute_f1_score(y_test, y_test_prob)
    print(f"F1 score: {f1score}")
    y_val_prob = best_model.predict(X_val)
    f1score_val = compute_f1_score(y_val, y_val_prob)
    print(f"F1 score: {f1score_val}")


def data_percentage(y):
    m = len(y)
    n_positive = np.sum(y)
    n_negative = m - n_positive

    pos_prec = (n_positive * 100.0) / m
    neg_prec = (n_negative * 100.0) / m

    print(f"Number of examples: {m}")
    print(f"Percentage of positive examples: {pos_prec}%, number of pos examples: {n_positive}")
    print(f"Percentage of negative examples: {neg_prec}%, number of neg examples: {n_negative}")
#data_percentage(y)

# the whole data


#print("Training Data:")
#data_percentage(y_train)
#print("Validation Data:")
#data_percentage(y_val)
#print("Testing Data:")
#data_percentage(y_test)

def getAccuracy(model):
    X,y=prepare_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, 0.2)
    result=model.evaluate(X_test, y_test, batch_size=32)
    return result[1]

def load_models():
    model=load_model(filepath='saved_model/cnn-parameters-improvement-23-0.91.model')
    return model


def predictTumourPresence(classifier,img_path):
    single_image = cv2.imread(img_path)
    single_image=crop_brain_contour(single_image, False)
    single_image = cv2.resize(single_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    single_image = image.img_to_array(single_image, data_format='channels_last')
    single_image = np.expand_dims(single_image, axis=0)

    output_array = classifier.predict(single_image)

    if output_array[0][0] == 1:
        predicted_output = True

    else:
        predicted_output = False

    return predicted_output

def plot_roc(top):
    model = load_models()
    # print(model.summary())
    X, y = prepare_data()
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y, test_size=0.2)

    prediction = model.predict(x_test)

    fpr, tpr, threshhold = roc_curve(y_test, prediction)

    roc_auc = auc(fpr, tpr)

    createWindowforGraph1(top, fpr, tpr, roc_auc)

    # plt.figure()
    #
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    #
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #
    # plt.xlim([0.0, 1.0])
    #
    # plt.ylim([0.0, 1.05])
    #
    # plt.xlabel('False Positive Rate')
    #
    # plt.ylabel('True Positive Rate')
    #
    # plt.title('ROC Curve')
    #
    # plt.legend(loc="lower right")
    #
    # plt.show()

if __name__ == '__main__':
    print("hi 1")


