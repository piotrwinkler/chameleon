from image_colorization.outdated.data_server import load_dataset
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, BatchNormalization, Reshape
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from time import time

dataset_path = "datasets/broadleaf/"


def main():
    x_train, y_train = load_dataset(dataset_path)

    # Network parameters
    batch_size = 16
    epochs = 100
    saving_period = 10
    learning_rate = 0.001
    model, checkpointer, tensorboard = prepare_network(learning_rate, saving_period,
                                                       'model_states/model_states.{epoch:02d}.hdf5', 'logs')
    model.save("model_states/modelV2.hdf5")

    model.fit(x_train, y_train,
              batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False,
              callbacks=[checkpointer, tensorboard])


def prepare_network(learning_rate, saving_period, models_dir, logs_dir):
    """
    This module prepares convolutional neural network model for future training.

    :param shape:
        Height/width of processed images.

    :param learning_rate:
        Length of single step in gradient propagation.

    :param saving_period:
        How many epochs of training must pass before saving model.

    :param models_dir:
        Models saving directory.

    :param logs_dir:
        Logs saving directory.

    :return:
        Randomly initialised model prepared for training.
        Checkpointer object containing parameters needed in model saving.
        Tensorboard object containing parameters needed in logs saving.
    """
    model = Sequential()

    model.add(Convolution2D(4, (5, 5), activation='relu', input_shape=(64, 64, 1)))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(8, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(16, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(64, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(128, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(32, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(8, (5, 5), activation='relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Convolution2D(1, (5, 5), activation='relu'))
    # model.add(BatchNormalization(axis=3))

    model.add(Flatten())
    model.add(Dense(8192, activation='relu'))
    model.add(Reshape((64, 64, 2)))


    # model.add(Convolution2D(32, (5, 5), activation='relu', padding='same'))
    # model.add(Convolution2D(32, (5, 5), activation='relu', padding='same'))
    # model.add(Convolution2D(2, (5, 5), activation='relu', padding='same'))

    # model.add(UpSampling2D(size=(2, 2)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(2, activation='softmax'))

    opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=models_dir,
                                   verbose=1, save_best_only=False, save_weights_only=False, period=saving_period)

    tensorboard = TensorBoard(log_dir=logs_dir.format(time()))

    return model, checkpointer, tensorboard


if __name__== "__main__":
    main()

