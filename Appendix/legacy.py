
"""
Utility functions to avoid warnings while testing both Keras 1 and 2.

legacy  ['legəsɪ] n. 遗赠，遗产
"""
import keras

keras_2 = int(keras.__version__.split(".")[0]) > 1  # Keras > 1

"""
fit_generator： 

Fits the model on data yielded batch-by-batch by a Python generator.

epoch  ['ɛpək] n. [地质] 世；新纪元；新时代；时间上的一点
"""
def fit_generator(model, generator, epochs, steps_per_epoch):
    """

    :param model:
    :param generator:
    :param epochs:
    :param steps_per_epoch:
    :return:

    Arguments:
        generator: A generator or an instance of `Sequence`
          (`keras.utils.Sequence`)
            object in order to avoid duplicate data
            when using multiprocessing.
            The output of the generator must be either
            - a tuple `(inputs, targets)`
            - a tuple `(inputs, targets, sample_weights)`.
            This tuple (a single output of the generator) makes a single batch.
            Therefore, all arrays in this tuple must have the same length (equal
            to the size of this batch). Different batches may have different
              sizes.
            For example, the last batch of the epoch is commonly smaller than
              the
            others, if the size of the dataset is not divisible by the batch
              size.
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
        steps_per_epoch: Total number of steps (batches of samples)
            to yield from `generator` before declaring one epoch
            finished and starting the next epoch. It should typically
            be equal to the number of samples of your dataset
            divided by the batch size.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        epochs: Integer, total number of iterations on the data.
    """
    if keras_2:
        model.fit_generator(generator, epochs=epochs, steps_per_epoch=steps_per_epoch)
    else:
        model.fit_generator(generator, nb_epoch=epochs, samples_per_epoch=steps_per_epoch)

"""
fit
"""
def fit(model, x, y, nb_epoch=10, *args, **kwargs):
    if keras_2:
        return model.fit(x, y, *args, epochs=nb_epoch, **kwargs)
    else:
        return model.fit(x, y, *args, nb_epoch=nb_epoch, **kwargs)


def l1l2(l1=0, l2=0):
    if keras_2:
        return keras.regularizers.L1L2(l1, l2)
    else:
        return keras.regularizers.l1l2(l1, l2)


def Dense(units, W_regularizer=None, W_initializer='glorot_uniform', **kwargs):
    if keras_2:
        return keras.layers.Dense(units, kernel_regularizer=W_regularizer, kernel_initializer=W_initializer, **kwargs)
    else:
        return keras.layers.Dense(units, W_regularizer=W_regularizer, init=W_initializer, **kwargs)


def BatchNormalization(mode=0, **kwargs):
    if keras_2:
        return keras.layers.BatchNormalization(**kwargs)
    else:
        return keras.layers.BatchNormalization(mode=mode, **kwargs)


def Convolution2D(units, w, h, W_regularizer=None, W_initializer='glorot_uniform', border_mode='same', **kwargs):
    if keras_2:
        return keras.layers.Convolution2D(units, (w, h), padding=border_mode, kernel_regularizer=W_regularizer,
                                          kernel_initializer=W_initializer,
                                          **kwargs)
    else:
        return keras.layers.Convolution2D(units, w, h, border_mode=border_mode, W_regularizer=W_regularizer,
                                          init=W_initializer,
                                          **kwargs)


def AveragePooling2D(pool_size, border_mode='valid', **kwargs):
    if keras_2:
        return keras.layers.AveragePooling2D(pool_size=pool_size, padding=border_mode, **kwargs)
    else:
        return keras.layers.AveragePooling2D(pool_size=pool_size, border_mode=border_mode, **kwargs)
