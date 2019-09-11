
#%%
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
K.set_image_data_format = 'channels_last'
#%%

# Dilated DenseNet

def DilatedDenseNet2D(in_shape, k, ls, theta, k_0, lbda=0, out_res=None,
               feed_pos=False, pos_noise_stdv=0):
    in_ = Input(shape=in_shape, name='input_X')

    if feed_pos:
        in_pos = Input(shape=(3,), name='input_position')
        pos = Reshape(target_shape=(1, 1, 3))(in_pos)
        if pos_noise_stdv != 0:
            pos = GaussianNoise(pos_noise_stdv)(pos)
        pos = BatchNormalization()(pos)

    x = Conv2D(filters=k_0, kernel_size=(7, 7), strides=(2, 2), padding='same')(in_)
    shortcuts = []
    for l in ls:
        x = denseBlock(mode='2D', l=l, k=k, lbda=lbda)(x)
        shortcuts.append(x)
        k_0 = int(round((k_0 + k * l) * theta))
        x = transitionLayerPool(mode='2D', f=k_0, lbda=lbda)(x)

    if feed_pos:
        shape = x._keras_shape[1:3]
        pos = UpSampling2D(size=shape)(pos)
        x = Concatenate(axis=-1)([x, pos])

    for l, shortcut in reversed(list(zip(ls, shortcuts))):
        x = denseBlock(mode='2D', l=l, k=k, lbda=lbda)(x)
        k_0 = int(round((k_0 + k * l) * theta / 2))
        x = transitionLayerUp(mode='2D', f=k_0, lbda=lbda)(x)
        x = Concatenate(axis=-1)([shortcut, x])
    x = UpSampling2D()(x)

    if out_res is not None:
        resize = resize_2D(out_res=out_res)(x)
        cut_in = Cropping2D(2 * ((in_shape[1] - out_res) // 2,))(in_)
        x = Concatenate(axis=-1)([cut_in, resize])

    x = Conv2D(filters=3, kernel_size=(1, 1))(x)
    out = Activation('softmax', name='output_Y')(x)
    if feed_pos:
        model = Model([in_, in_pos], out)
    else:
        model = Model(in_, out)
    return model



       
   
class resize_2D(Layer):
    def __init__(self, out_res=24, **kwargs):
        self.input_dim = None
        self.out_res = out_res
        super(resize_2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1:]
        super(resize_2D, self).build(input_shape)  

    def call(self, x):
        size = (K.constant(np.array(2 * (self.out_res,), dtype=np.int32), dtype=K.tf.int32))
        y = K.tf.image.resize_bilinear(images=x, size=size)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + 2 * (self.out_res,) + (input_shape[-1],)





def denseBlock(mode, l, k, lbda):
    if mode == '2D':
        def dense_block_instance(x):
            ins = [x, denseConv('2D', k, 3, lbda)(
                denseConv('2D', k, 1, lbda)(x))]
            for i in range(l - 1):
                ins.append(denseConv('2D', k, 3, lbda)(
                    denseConv('2D', k, 1, lbda)(Concatenate(axis=-1)(ins))))
            y = Concatenate(axis=-1)(ins)
            return y

        return dense_block_instance
    else:
        def dense_block_instance(x):

            ins = [x, denseConv('3D', k, 3, lbda)(
                denseConv('3D', k, 1, lbda)(x))]  
            for i in range(l - 1):
                ins.append(denseConv('3D', k, 3, lbda)(
                    denseConv('3D', k, 1, lbda)(Concatenate(axis=-1)(
                        ins)))) 
            y = Concatenate(axis=-1)(ins) 
            return y  

        return dense_block_instance


def denseConv(mode, k, kernel_size, lbda, padding = 'same',dilation = 1):
    if mode == '2D':
        return lambda x: Conv2D(filters=k,
                                kernel_size=2 * (kernel_size,),
                                padding='same',
                                kernel_regularizer=regularizers.l2(lbda),
                                bias_regularizer=regularizers.l2(lbda))(
            Activation('relu')(
                BatchNormalization()(x)))
    else:
        return lambda x: Conv3D(filters=k,
                                kernel_size=3 * (kernel_size,), 
                                padding= padding,
                                kernel_regularizer=regularizers.l2(lbda), 
                                bias_regularizer=regularizers.l2(lbda),
                                dilation_rate= 3 * (dilation,))(
            Activation('relu')(
                BatchNormalization()(x)))



def transitionLayerPool(mode, f, lbda):   
    if mode == '2D':
        return lambda x: AveragePooling2D(pool_size=2*(2,))(
                         denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: AveragePooling3D(pool_size=3*(2,))(   
                         denseConv('3D', f, 1, lbda)(x))


def transitionLayerPool0(mode, f, lbda):  
    if mode == '2D':
        return lambda x: AveragePooling2D(pool_size=2 * (2,))(
            denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: denseConv('3D', f, 3, lbda, 'valid', 3)(denseConv('3D', f, 3, lbda, 'valid', 2)(
            denseConv('3D', f, 3, lbda, 'valid', 2)(denseConv('3D', f, 3, lbda, 'valid', 1)(
                (denseConv('3D', f, 3, lbda, 'same',1)(x))))))

def transitionLayerPool1(mode, f, lbda): 
    if mode == '2D':
        return lambda x: AveragePooling2D(pool_size=2 * (2,))(
            denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: denseConv('3D', f, 3, lbda, 'valid', 3)(denseConv('3D', f, 3, lbda, 'valid', 1)(
                (denseConv('3D', f, 3, lbda, 'same',1)(x))))

def transitionLayerPool2(mode, f, lbda):  
    if mode == '2D':
        return lambda x: AveragePooling2D(pool_size=2 * (2,))(
            denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: denseConv('3D', f, 3, lbda, 'valid', 2)(
            (denseConv('3D', f, 3, lbda, 'same', 1)(x)))

def transitionLayerPool3(mode, f, lbda): 
    if mode == '2D':
        return lambda x: AveragePooling2D(pool_size=2 * (2,))(
            denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: denseConv('3D', f, 3, lbda, 'valid', 1)(
            (denseConv('3D', f, 3, lbda, 'same', 1)(x)))

def transitionLayerPool4(mode, f, lbda):  
    if mode == '2D':
        return lambda x: AveragePooling2D(pool_size=2 * (2,))(
            denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: denseConv('3D', f, 2, lbda, 'valid', 1)(
            (denseConv('3D', f, 1, lbda, 'same', 1)(x)))

def transitionLayerUp(mode, f, lbda):
    if mode == '2D':
        return lambda x: UpSampling2D()(
            denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: UpSampling3D()(
            denseConv('3D', f, 1, lbda)(x))

#%%
model=DilatedDenseNet2D(in_shape=(512,512,2), k=16, ls=[8,8], theta=0.5, k_0=32)
model.summary()
#%%
plot_model(model, to_file='model.png')


#%%
