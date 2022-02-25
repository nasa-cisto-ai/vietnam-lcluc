import logging
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import concatenate, Input, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose
from tensorflow.keras.regularizers import l2


__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ---------------------------------------------------------------------------
# module unet
#
# Build UNet NN architecture using Keras. Any of these functions can be
# called from an external script. UNets can be modified as needed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


# --------------------------- Convolution Functions ----------------------- #

def standard_unit(
    input_tensor, stage, nb_filter, act="relu", dropout_rate=0.5, kernel_size=3
     ):

    x = Conv2D(
        nb_filter, (kernel_size, kernel_size), activation=act,
        name='conv' + stage + '_1', kernel_initializer='he_normal',
        padding='same', kernel_regularizer=l2(1e-4)
    )(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(
        nb_filter, (kernel_size, kernel_size), activation=act,
        name='conv' + stage + '_2', kernel_initializer='he_normal',
        padding='same', kernel_regularizer=l2(1e-4)
    )(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x


def unet_dropout(nclass=19, input_size=(256, 256, 6), weight_file=None,
                 maps=[64, 128, 256, 512, 1024]
                 ):
    """
    UNet network using dropout features.
    """
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c4)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)

    # Squeeze
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(c5)
    d5 = Dropout(0.5)(c5)

    # Decoder
    u6 = UpSampling2D(size=(2, 2))(d5)
    m6 = concatenate([d4, u6], axis=3)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(m6)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D(size=(2, 2))(c6)
    m7 = concatenate([c3, u7], axis=3)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(m7)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D(size=(2, 2))(c7)
    m8 = concatenate([c2, u8], axis=3)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(m8)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D(size=(2, 2))(c8)
    m9 = concatenate([c1, u9], axis=3)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(m9)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c9)

    actv = 'softmax'
    if nclass == 1:
        actv = 'sigmoid'

    conv10 = Conv2D(nclass, (1, 1), activation=actv)(c9)
    model = Model(inputs=inputs, outputs=conv10, name="UNetDropout")

    if weight_file:
        model.load_weights(weight_file)
    return model


def unet_batchnorm(nclass=19, input_size=(256, 256, 6), weight_file=None,
                   kr=l2(0.0001), maps=[64, 128, 256, 512, 1024]
                   ):
    """
    UNet network using batch normalization features.
    """
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c1)
    n1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(n1)

    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c2)
    n2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(n2)

    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c3)
    n3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(n3)

    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c4)
    n4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(n4)

    # Squeeze
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(maps[4], (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    n6 = BatchNormalization()(u6)
    u6 = concatenate([n6, n4])
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(maps[3], (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    n7 = BatchNormalization()(u7)
    u7 = concatenate([n7, n3])
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(maps[2], (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    n8 = BatchNormalization()(u8)
    u8 = concatenate([n8, n2])
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(maps[1], (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    n9 = BatchNormalization()(u9)
    u9 = concatenate([n9, n1], axis=3)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(maps[0], (3, 3), activation='relu', padding='same')(c9)

    actv = 'softmax'
    if nclass == 1:
        actv = 'sigmoid'

    c10 = Conv2D(nclass, (1, 1), activation=actv, kernel_regularizer=kr)(c9)
    model = Model(inputs=inputs, outputs=c10, name="UNetBatchNorm")

    if weight_file:
        model.load_weights(weight_file)
    return model


def w_unet(
    input_size=(256, 256, 6),
    num_class=1, nb_filter=[32, 64, 128, 256, 512]
):
    """
    wU-Net for comparison
    Total params: 9,282,246
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=input_size, name='main_input')
    else:
        bn_axis = 1
        img_input = Input(shape=input_size, name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(inputs=img_input, outputs=unet_output)

    return model


def unet_plus_plus(
    img_rows=256, img_cols=256, color_type=1, num_class=1,
    nb_filter=[32, 64, 128, 256, 512], deep_supervision=False
     ):
    """
    Standard UNet++ [Zhou et.al, 2018]
    Total params: 9,041,601
    """
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='m_input')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='m_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name='up12',
        padding='same'
    )(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name='up22',
        padding='same'
    )(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name='up13',
        padding='same'
    )(conv2_2)
    conv1_3 = concatenate(
        [up1_3, conv1_1, conv1_2], name='merge13',
        axis=bn_axis
    )
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(
        nb_filter[2], (2, 2), strides=(2, 2), name='up32',
        padding='same'
    )(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name='up23',
        padding='same'
    )(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name='up14',
        padding='same'
    )(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(
        nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same'
    )(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(
        nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same'
    )(conv4_2)
    conv3_3 = concatenate(
        [up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis
    )
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(
        nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same'
    )(conv3_3)
    conv2_4 = concatenate(
        [up2_4, conv2_1, conv2_2, conv2_3], name='merge24',
        axis=bn_axis
    )
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(
        nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same'
    )(conv2_4)
    conv1_5 = concatenate(
        [up1_5, conv1_1, conv1_2, conv1_3, conv1_4],
        name='merge15', axis=bn_axis
    )
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(
        num_class, (1, 1), activation='sigmoid', name='output_1',
        kernel_initializer='he_normal', padding='same',
        kernel_regularizer=l2(1e-4)
    )(conv1_2)
    nestnet_output_2 = Conv2D(
        num_class, (1, 1), activation='sigmoid', name='output_2',
        kernel_initializer='he_normal', padding='same',
        kernel_regularizer=l2(1e-4)
    )(conv1_3)
    nestnet_output_3 = Conv2D(
        num_class, (1, 1), activation='sigmoid', name='output_3',
        kernel_initializer='he_normal', padding='same',
        kernel_regularizer=l2(1e-4)
    )(conv1_4)
    nestnet_output_4 = Conv2D(
        num_class, (1, 1), activation='sigmoid', name='output_4',
        kernel_initializer='he_normal', padding='same',
        kernel_regularizer=l2(1e-4)
    )(conv1_5)

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(inputs=img_input, outputs=[nestnet_output_4])

    return model

"""
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

#Convolutional block to be used in autoencoder and U-Net
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block for autoencoder (no skip connections)
def decoder_block(input, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = conv_block(x, num_filters)
    return x

#Encoder will be the same for Autoencoder and U-net
#We are getting both conv output and maxpool output for convenience.
#we will ignore conv output for Autoencoder. It acts as skip connections for U-Net
def build_encoder(input_image):
    #inputs = Input(input_shape)

    s1, p1 = encoder_block(input_image, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    encoded = conv_block(p4, 1024) #Bridge
    
    return encoded

#Decoder for Autoencoder ONLY. 
def build_decoder(encoded):
    d1 = decoder_block(encoded, 512)
    d2 = decoder_block(d1, 256)
    d3 = decoder_block(d2, 128)
    d4 = decoder_block(d3, 64)
    
    decoded = Conv2D(3, 3, padding="same", activation="sigmoid")(d4)
    return decoded

#Use encoder and decoder blocks to build the autoencoder. 
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    autoencoder = Model(input_img, build_decoder(build_encoder(input_img)))
    return(autoencoder)

# model=build_autoencoder((256, 256, 3))
# print(model.summary())

#Decoder block for unet
#skip features gets input from encoder for concatenation
def decoder_block_for_unet(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block_for_unet(b1, s4, 512)
    d2 = decoder_block_for_unet(d1, s3, 256)
    d3 = decoder_block_for_unet(d2, s2, 128)
    d4 = decoder_block_for_unet(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    print(model.summary())
    return model
"""

# -------------------------------------------------------------------------------
# module unet Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Can add different images sizes - for now: (256,256,6)
    simple_unet = unet_dropout()
    simple_unet.summary()

    # Batch Normalization UNet
    simple_unet = unet_batchnorm()
    simple_unet.summary()

    # W-UNet
    simple_unet = w_unet()
    simple_unet.summary()

    # UNet Plus Plus
    simple_unet = unet_plus_plus()
    simple_unet.summary()
