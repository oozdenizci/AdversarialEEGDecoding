#!/usr/bin/env python
import keras
import keras.backend as tf
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D
from keras.layers.convolutional import AveragePooling2D, Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.constraints import max_norm
import numpy as np
keras.backend.set_image_data_format('channels_last')


class AdversarialCNN:
    def __init__(self, chans, samples, n_output, n_nuisance, architecture='DeepConvNet', adversarial=False, lam=0.01):

        # Input, data set and model training scheme parameters
        self.chans = chans
        self.samples = samples
        self.n_output = n_output
        self.n_nuisance = n_nuisance
        self.lam = lam

        # Build the network blocks
        self.enc = self.encoder_model(architecture)
        self.latent_dim = self.enc.output_shape[-1]  # inherit latent dimensionality
        self.cla = self.classifier_model(architecture)
        self.adv = self.adversary_model()

        # Compile the network with or without adversarial censoring
        input = Input(shape=(self.chans, self.samples, 1))
        latent = self.enc(input)
        output = self.cla(latent)
        leakage = self.adv(latent)
        self.adv.trainable = False
        self.acnn = Model(input, [output, leakage])

        if adversarial:
            self.acnn.compile(loss=[lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                                    lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True)],
                              loss_weights=[1., -1. * self.lam], optimizer=Adam(lr=1e-3, decay=1e-4),
                              metrics=['accuracy'])
        else:   # trains a regular (non-adversarial) CNN, but will monitor leakage via the adversary alongside
            self.acnn.compile(loss=[lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                                    lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True)],
                              loss_weights=[1., 0.], optimizer=Adam(lr=1e-3, decay=1e-4),
                              metrics=['accuracy'])

        self.adv.trainable = True
        self.adv.compile(loss=lambda x, y: tf.categorical_crossentropy(x, y, from_logits=True),
                         loss_weights=[self.lam],
                         optimizer=Adam(lr=1e-3, decay=1e-4),
                         metrics=['accuracy'])

    def encoder_model(self, architecture):
        model = Sequential()
        if architecture == 'EEGNet':
            model.add(Conv2D(8, (1, 32), padding='same', use_bias=False))
            model.add(BatchNormalization(axis=3))
            model.add(DepthwiseConv2D((self.chans, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.)))
            model.add(BatchNormalization(axis=3))
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1, 4)))
            model.add(Dropout(0.25))
            model.add(SeparableConv2D(16, (1, 16), use_bias=False, padding='same'))
            model.add(BatchNormalization(axis=3))
            model.add(Activation('elu'))
            model.add(AveragePooling2D((1, 8)))
            model.add(Dropout(0.25))
            model.add(Flatten())
        elif architecture == 'DeepConvNet':
            model.add(Conv2D(25, (1, 5)))
            model.add(Conv2D(25, (self.chans, 1), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(50, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(100, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(200, (1, 5), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation('elu'))
            model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
        elif architecture == 'ShallowConvNet':
            model.add(Conv2D(40, (1, 13)))
            model.add(Conv2D(40, (self.chans, 1), use_bias=False))
            model.add(BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1))
            model.add(Activation(lambda x: tf.square(x)))
            model.add(AveragePooling2D(pool_size=(1, 35), strides=(1, 7)))
            model.add(Activation(lambda x: tf.log(tf.clip(x, min_value=1e-7, max_value=10000))))
            model.add(Dropout(0.5))
            model.add(Flatten())

        input = Input(shape=(self.chans, self.samples, 1))
        latent = model(input)

        return Model(input, latent, name='enc')

    def classifier_model(self, architecture):
        latent = Input(shape=(self.latent_dim,))
        if architecture == 'EEGNet':
            output = Dense(self.n_output, kernel_constraint=max_norm(0.25))(latent)
        elif architecture == 'DeepConvNet' or architecture == 'ShallowConvNet':
            output = Dense(self.n_output)(latent)

        return Model(latent, output, name='cla')

    def adversary_model(self):
        latent = Input(shape=(self.latent_dim,))
        leakage = Dense(self.n_nuisance)(latent)

        return Model(latent, leakage, name='adv')

    def train(self, train_set, val_set, log, epochs=500, batch_size=40):
        x_train, y_train, s_train = train_set
        x_test, y_test, s_test = val_set

        train_index = np.arange(y_train.shape[0])
        train_batches = [(i * batch_size, min(y_train.shape[0], (i + 1) * batch_size))
                         for i in range((y_train.shape[0] + batch_size - 1) // batch_size)]

        # Early stopping variables
        es_wait = 0
        es_best = np.Inf
        es_best_weights = None

        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            np.random.shuffle(train_index)
            train_log = []
            for iter, (batch_start, batch_end) in enumerate(train_batches):
                batch_ids = train_index[batch_start:batch_end]
                x_train_batch = x_train[batch_ids]
                y_train_batch = y_train[batch_ids]
                s_train_batch = s_train[batch_ids]
                z_train_batch = self.enc.predict_on_batch(x_train_batch)

                self.adv.train_on_batch(z_train_batch, s_train_batch)
                train_log.append(self.acnn.train_on_batch(x_train_batch, [y_train_batch, s_train_batch]))
            train_log = np.mean(train_log, axis=0)
            val_log = self.acnn.test_on_batch(x_test, [y_test, s_test])

            # Logging model training information per epoch
            print("Train - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%]"
                  % (train_log[0], train_log[1], 100*train_log[3], train_log[2], 100*train_log[4]))
            print("Validation - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%]"
                  % (val_log[0], val_log[1], 100*val_log[3], val_log[2], 100*val_log[4]))
            with open(log + '/train.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(train_log[0]) + ',' + str(train_log[1]) + ',' +
                        str(100*train_log[3]) + ',' + str(train_log[2]) + ',' + str(100*train_log[4]) + '\n')
            with open(log + '/validation.csv', 'a') as f:
                f.write(str(epoch) + ',' + str(val_log[0]) + ',' + str(val_log[1]) + ',' +
                        str(100*val_log[3]) + ',' + str(val_log[2]) + ',' + str(100*val_log[4]) + '\n')

            # Check early stopping criteria based on validation CLA loss - patience for 10 epochs
            if np.less(val_log[1], es_best):
                es_wait = 0
                es_best = val_log[1]
                es_best_weights = self.acnn.get_weights()
            else:
                es_wait += 1
                if es_wait >= 10:
                    print('Early stopping...')
                    self.acnn.set_weights(es_best_weights)
                    return
