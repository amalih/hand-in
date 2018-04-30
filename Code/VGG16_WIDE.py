
# This model uses the VGG16 pretrained base, and has "wide" top layers of 1000 + 200 nodes.

from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout, Input, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras import backend as K

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16_WIDE():
    # Create the base pre-trained model
    K.set_image_dim_ordering('tf')

    # Building VGG16 base:
    img_input = Input(shape=(64, 64, 3))

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    base_model = Model(img_input, x, name='vgg16')

    # load weights
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models', file_hash='6d6bbae143d832006294945121d1f1fc')
    base_model.load_weights(weights_path)

    x = base_model.output
    x = Flatten()(x)

    # Regression layers
    x = Dense(1000, activation='relu', name='fc1', W_regularizer=l2(0.0001))(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation='relu', name='fc2', W_regularizer=l2(0.0001))(x)
    
    predictions = Dense(1)(x)

    model = Model(input=base_model.input, output=predictions)

    # Train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='mean_squared_error', optimizer='adam', metric=['mse'])

    model.save('VGG16_WIDE.h5')

    return 0;
