####################
# TreatmentResponsenet is modified from https://gitlab.com/iss_mia/nako/age_estimation
######################
#%%
from tensorflow.python.client import device_lib

import tensorflow.keras.backend as K

from tensorflow.keras.layers import *

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Activation,
    Lambda,
    concatenate,
    Reshape,
    Dropout,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    add,
    Maximum
)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential

import numpy as np
#%%
# create the model with self-defined class
class BasicModel:

    # the layers are set as the attributes of this class
    def __init__(self, depth=10, k=4, num_slice_per_group=40):
        self._depth = depth
        self._k = k
        self.num_slice_per_group = num_slice_per_group
        self._dropout_probability = 0.5
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"
        self._channel_axis = -1

        self.conv1 = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                            padding='valid',
                            strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                            padding='valid',
                            strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')

        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                            padding='same',
                            strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')

        self.conv4 = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                            padding='same',
                            strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')

        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv5 = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                            padding='same',
                            strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')
        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                            padding='same',
                            strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')
        self.conv7 = Conv2D(filters=256, kernel_size=(3, 3), kernel_initializer='he_normal', weights=None,
                            padding='same',
                            strides=(1, 1), kernel_regularizer=l2(1e-6), activation='relu')

        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.globalaveragepooling = GlobalAveragePooling2D()

        return

        # Wide residual network http://arxiv.org/abs/1605.07146
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                print(i)
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                   strides=v[2],
                                   padding=v[3],
                                   kernel_initializer=self._weight_init,
                                   kernel_regularizer=l2(self._weight_decay),
                                   use_bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                   strides=v[2],
                                   padding=v[3],
                                   kernel_initializer=self._weight_init,
                                   kernel_regularizer=l2(self._weight_decay),
                                   use_bias=self._use_bias)(convs)

            # Shortcut Connection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),
                                  strides=stride,
                                  padding="same",
                                  kernel_initializer=self._weight_init,
                                  kernel_regularizer=l2(self._weight_decay),
                                  use_bias=self._use_bias)(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f

        # "Stacking Residual Units on the same stage"
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

    # the input shall be input tensor, which is a 2D slice tensor in the reality
    def _build_basic_network(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.pool1(x)

        x1 = self.conv5(out)
        x = self.conv6(out)
        x = self.conv7(x)
        x = add([x1, x])
        x = self.pool2(x)
        x = self.globalaveragepooling(x)

        return x

    def _build_basic_network_widecnn(self, input_tensor):
        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),
                       strides=(1, 1),
                       padding="same",
                       kernel_initializer=self._weight_init,
                       kernel_regularizer=l2(self._weight_decay),
                       use_bias=self._use_bias)(input_tensor)  # "One conv at the beginning (spatial size: 32x32)"

        # Add wide residual blocks
        block_fn = self._wide_basic

        conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(
            conv1)

        conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(
            conv2)

        conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(
            conv3)

        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        # Classifier block
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
        flatten_1 = GlobalAveragePooling2D()(pool)

        return flatten_1

#########################
# Input: patchSize (image size of the slice in the group)
# Output: TreatmentResponseNet model
# num_slice_per_group: the number of slices are extracted in a sample
##########################

def createModel(patchSize, num_slice_per_group):

    input_tensor = Input(shape=(patchSize[0], patchSize[1],num_slice_per_group))
    input_gender = Input(shape=(1,), name='gender_input')
    input_weight = Input(shape=(1,), name='weight_input')
    input_height = Input(shape=(1,), name='height_input')
    input_age = Input(shape=(1,), name='age_input')
    input_BMI = Input(shape=(1,), name='BMI_input')
    input_VOI = Input(shape=(1,), name='VOI_input')
    input_SUV = Input(shape=(1,),name='SUV_input')
    input_Orts = Input(shape=(1,), name='Orts_input')
    def get_slice (x, index):
        return x[:, :, :, index]

    input_list = []

    # create the input tensor list
    for i in range(num_slice_per_group):
        sub_input = Lambda(get_slice, output_shape=(patchSize[0], patchSize[1],1), arguments={'index': i})(
            input_tensor)
        sub_input = Reshape((patchSize[0], patchSize[1],1))(sub_input)
        input_list.append(sub_input)

    # instance the basic model
    basic_model = BasicModel()

    output_list =[]

    # use basic network for each input slice, and append their output
    for input in input_list:
        output = basic_model._build_basic_network(input)
        output_list.append(output)

    # concatenate image and metadata
    
    x1 = concatenate([output_list[0], input_gender, input_height, input_weight, input_age,input_BMI,input_VOI,input_SUV,input_Orts])
    x2 = concatenate([output_list[1], input_gender, input_height, input_weight, input_age,input_BMI,input_VOI,input_SUV,input_Orts])
    x=concatenate([x1,x2])
    
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = Dropout(0.5)(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)

    # fully-connected layer
    output_response = Dense(units=4,
                   activation='softmax',
                   name='Response_Classification')(x)

    # output_survivalRate = Dense(units=1,
    #                activation='linear',
    #                name='Survival_Rate')(x)

    output_survivalRate = Dense(units=3,
                    activation='softmax',
                    name='Survival_Rate')(x)

    # output_treatmentRegress = Dense(units=1,
    #                activation='linear',
    #                name='Treatment_Regress')(x)

    output_treatmentRegress = Dense(units=3,
                    activation='softmax',
                    name='Treatment_Regress')(x)

    sModelName = 'ResponseNet'
    cnn = Model([input_tensor,input_age,input_gender,input_weight,input_height,input_BMI,input_VOI,input_SUV,input_Orts],[output_response,output_survivalRate,output_treatmentRegress],name=sModelName)
    return cnn, sModelName