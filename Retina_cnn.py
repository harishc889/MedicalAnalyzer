from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import os.path
import tkinter.messagebox as mb

nb_train_samples = 10000  # 2541
nb_validation_samples = 2000  # 613
batch_size = 16 # 16


# def helloCallBack():
#   mb.showinfo( "Hello Python", "Hello World")
#   return 1

def buildClassifier():
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3),input_shape=(256, 256, 1), activation='relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    ## Adding a third convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    ## Adding a third convolutional layer
    # classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Flattening
    classifier.add(Flatten())

    # Full connection
    classifier.add(Dense(units=512, activation='relu'))
    # classifier.add(Dense(units = 1024, activation = 'relu'))
    # classifier.add(Dense(units = 512, activation = 'relu'))
    classifier.add(Dropout(0.1))

    classifier.add(Dense(units=512,  activation='relu'))
    # classifier.add(Dropout(0.1))

    classifier.add(Dense(units=512,  activation='relu'))
    classifier.add(Dropout(0.05))

    classifier.add(Dense(units=4, activation='softmax'))

    # Compiling the CNN
    from keras.optimizers import RMSprop, SGD
    # classifier.compile(loss = 'categorical_crossentropy',
    #              optimizer = RMSprop(lr = 0.001),
    #              metrics = ['accuracy'])

    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # tkMessageBox.showinfo( "Hello Python", "Hello World")
    return classifier


# Fitting the CNN to the images


def dataGenerator():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('OCT2017/train',
                                                     target_size=(256, 256),
                                                     batch_size=16,
                                                     class_mode='categorical',
                                                     color_mode='grayscale',
                                                     shuffle=True)

    test_set = test_datagen.flow_from_directory('OCT2017/test',
                                                target_size=(256, 256),
                                                batch_size=16,
                                                class_mode='categorical',
                                                color_mode='grayscale',
                                                shuffle=False)
    return training_set, test_set


def fitClassifier(classifier, training_set, test_set):
    history = classifier.fit_generator(training_set,
                                       steps_per_epoch=nb_train_samples // batch_size,
                                       epochs=10,
                                       validation_data=test_set,
                                       validation_steps=nb_validation_samples // batch_size)
    return classifier, history


# from keras.models import load_model
# dataGenerator()
# fitClassifier(classifier)
# classifier.save('savedModel/retina_model') # creates a HDF5 file ‘my_model.h5’
# del model # deletes the existing model

def predictSingle(classifier, img_path):
    single_image = image.load_img(img_path, color_mode='grayscale', target_size=(256, 256, 1))
    single_image = image.img_to_array(single_image, data_format='channels_last')
    single_image = np.expand_dims(single_image, axis=0)

    output_array = classifier.predict(single_image)

    if output_array[0][0] == 1:
        predicted_output = 'CNV'

    elif output_array[0][1] == 1:
        predicted_output = 'DME'

    elif output_array[0][2] == 1:
        predicted_output = 'DRUSEN'
    else:
        predicted_output = 'NORMAL'

    return predicted_output


# predictSingle()


def loadHistory():
    history = pickle.load(open("use/RetinaHistoryDict1", "rb"))
    # print(history.keys())
    return history


def compareModel(new_model, test_set):
    # if os.path.isfile('savedModel/retina_model'):
    # flag=1
    old_model = load_model('use/retina_model.h5')
    score = old_model.evaluate_generator(generator=test_set,
                                         steps=nb_validation_samples // batch_size

                                         )
    new_score = new_model.evaluate_generator(generator=test_set,
                                             steps=nb_validation_samples // batch_size

                                             )
    if score[1] < new_score[1]:
        return True
    else:
        return False


# compareModel()
# saving model
def savingModel():
    classifier = buildClassifier()
    training_set, test_set = dataGenerator()
    classifier, history = fitClassifier(classifier, training_set, test_set)

    with open("use/RetinaHistoryDict1", "wb") as file_pi:
        pickle.dump(history.history, file_pi)

    if os.path.isfile('use/retina_model.h5'):

        decider = (compareModel(classifier, test_set))
        if decider == True:
            classifier.save('use/retina_model.h5')
            print('saved1')

    else:
        classifier.save('use/retina_model.h5')
        print('saved2')
    return history


# savingModel()
def getHistory(classifier, training_set, test_set):
    history = classifier.fit_generator(training_set,
                                       steps_per_epoch=nb_train_samples // batch_size,
                                       epochs=10,
                                       validation_data=test_set,
                                       validation_steps=nb_validation_samples // batch_size)
    return history


def retrieveModel():
    old_model = load_model('use/retina_model.h5')
    return old_model


def getAccuracy(old_model):
    old_model = retrieveModel()
    training_set, test_set = dataGenerator()
    score = old_model.evaluate_generator(generator=test_set,
                                         steps=nb_validation_samples // batch_size

                                         )
    return score[1]


def helloCallBack():
    # mb.showinfo( "Hello Python", "Hello World")

    return '1'


if __name__ == '__main__':
    print('hello')

    history = savingModel()
    print(history.history.keys())

    # print(training_set.class_indices)
    # savingModel()
    # model=retrieveModel()
    # print(model.summary())

    # =============================================================================
    #     history = pickle.load(open("saved_model/RetinaHistoryDict1","rb"))
    #     print(history.keys())
    # =============================================================================

    acc = history.history['accuracy']
    print(acc)
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(10)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
# =============================================================================
# =============================================================================
#     savingModel()
#     model=retrieveModel()
#     print(model.summary())
# =============================================================================
