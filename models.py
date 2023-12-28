from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, Reshape, Activation, Permute, Multiply
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Bidirectional, LSTM, GRU, RNN
from tensorflow.keras import Input, Model
from tensorflow import transpose, reshape, split
from tensorflow.keras.layers import LSTMCell, StackedRNNCells
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers
# from tensorflow.keras.regularizers import l2

def scheduler(current_epoch, learning_rate):
    """
    Lowers the learning rate hyperparameter relative to the number of epochs.
    """
    if current_epoch < 10: #2
        learning_rate = 0.001
    elif current_epoch < 195: #37
        learning_rate = 0.01
    else:
        learning_rate = 0.0001
    return learning_rate

def get_lr_metric(optimizer):
    """
    Returns the current learning rate.
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr

class SaveAtEpochEnd(Callback):
    """
    Saves the model after each N epochs.
    """
    def __init__(self, each_n_epochs, file_path):
        """
        Initialization function of the class.
        
        Parameters:
            - each_n_epochs: determines the number of epochs that needs to end for the model to be saved;
            - file_path: the model will be saved in a .h5 file with this path. The '.h5' part doesn't need to be
            included in this parameter.
        """
        # Intializing variables
        self.each_n_epochs = each_n_epochs
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs={}):
        if self.each_n_epochs == 1:
            self.model.save(self.file_path + '.h5')
            print(f'SaveAtEpochEnd: The model was saved in {self.file_path}.h5')

        elif epoch % self.each_n_epochs == 0:
            self.model.save(self.file_path + '.h5')
            print(f'SaveAtEpochEnd: The model was saved in {self.file_path}.h5')

def InceptionBlock(input_img, block_index, block_type='basic', filters_sizes=(64, 96, 128, 16, 32, 128, 32), factor=1):
    """
    Creates and returns an inception block for a CNN.

    Parameters:
        - input_img: input data for the inception block;
        - block_index: index of the inception block;
    
    Optional Parameters:
        - block_type: what type of inception block will be generated. Default value is 'basic';
        - filters_sizes: tuple of filter sizes for each of the 7 convolution layers of this inception block. Default
        tuple is (64, 96, 128, 16, 32, 128, 32);
        - factor: used to multiply the number of filters used in each convolution layer simultaneously. Default
        value is 1:
    """
    result = -1

    if(block_type == 'basic' or block_type == 'flat'):
        conv1_1_1 = Conv1D(int(filters_sizes[0] * factor), 1, padding='same', activation='relu', name=f'conv1_{block_index}_1_f{factor}')(input_img)
        conv2_1_1 = Conv1D(int(filters_sizes[1] * factor), 1, padding='same', activation='relu', name=f'conv2_{block_index}_1_f{factor}')(input_img)
        conv2_1_2 = Conv1D(int(filters_sizes[2] * factor), 5, padding='same', activation='relu', name=f'conv2_{block_index}_2_f{factor}')(conv2_1_1)
        conv3_1_1 = Conv1D(int(filters_sizes[3] * factor), 1, padding='same', activation='relu', name=f'conv3_{block_index}_1_f{factor}')(input_img)
        conv3_1_2 = Conv1D(int(filters_sizes[4] * factor), 3, padding='same', activation='relu', name=f'conv3_{block_index}_2_f{factor}')(conv3_1_1)
        conv4_1_1 = Conv1D(int(filters_sizes[5] * factor), 2, padding='same', activation='relu', name=f'conv4_{block_index}_1_f{factor}')(input_img)
        maxP_3_1 = MaxPooling1D(pool_size=3, strides=1, padding="same", name=f'maxP_3_{block_index}_f{factor}')(conv4_1_1)
        conv4_1_2 = Conv1D(int(filters_sizes[6] * factor), 1, padding='same', activation='relu', name=f'conv4_{block_index}_2_f{factor}')(maxP_3_1)

        result = Concatenate(axis=2)([conv1_1_1, conv2_1_2, conv3_1_2, conv4_1_2])

        # Generated Inception Block will have a flat output
        if(block_type == 'flat'):
            result = Flatten()(result)
    else:
        print('ERROR: Invalid Inception Block type.\n')

    return result

def SEBlock(input, block_type='basic', se_ratio = 16, activation = "relu", data_format = 'channels_last', ki = "he_normal"):
    '''
    Creates and returns a squeeze & excitation block for a CNN.

    Parameters:
        - input: input data for the squeeze & excitation block;
    Optional Parameters:
        - block_type: what type of squeeze & excitation block will be generated. Default value is 'basic';
        - se_ratio : ratio for reducing the number of filters in the first dense layer of the block. Default
        value is 16;
        - activation : activation function of the first dense layer. Default value is "relu";
        - data_format : if channel axis is the first dimension of the input, this parameter should be
        'channels_first', and if it's the last dimension, this parameter should be 'channels_last'. Default
        value is 'channels_last';
        - ki : kernel initializer. Default value is "he_normal".
    '''
    x = -1

    if(block_type == 'basic' or block_type == 'flat'):
        channel_axis = -1 if data_format == 'channels_last' else 1
        input_channels = input.shape[channel_axis]

        reduced_channels = input_channels // se_ratio

        # Squeeze operation
        x = GlobalAveragePooling1D()(input)
        x = Reshape(1,1,input_channels)(x) if data_format == 'channels_first' else x
        x = Dense(reduced_channels, kernel_initializer= ki)(x)
        x = Activation(activation)(x)

        # Excitation operation
        x = Dense(input_channels, kernel_initializer=ki, activation='sigmoid')(x)
        x = Permute(dims=(3,1,2))(x) if data_format == 'channels_first' else x
        x = Multiply()([input, x])

        # Generated Squeeze and Excitation Block will have a flat output
        if(block_type == 'flat'):
            x = Flatten()(x)
    else:
        print('ERROR: Invalid Squeeze and Excitation Block type.\n')

    return x

def create_model(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the CNN model.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """
    model = None

    if(remove_last_layer == False):
        model = Sequential(name='Biometric_for_Identification')
    else:
        model = Sequential(name='Biometric_for_Verification')

    # Conv1
    model.add(Conv1D(96, (11), input_shape=(window_size, num_channels), activation='relu', name='Conv1'))
    model.add(BatchNormalization(name='Norm1'))
    # Pool1
    model.add(MaxPooling1D(strides=4, name='Pool1'))
    # Conv2
    model.add(Conv1D(128, (9), activation='relu', name='Conv2'))
    model.add(BatchNormalization(name='Norm2'))
    # Pool2
    model.add(MaxPooling1D(strides=2, name='Pool2'))
    # Conv3
    model.add(Conv1D(256, (9), activation='relu', name='Conv3')) 
    model.add(BatchNormalization(name='Norm3'))
    # Pool3
    model.add(MaxPooling1D(strides=2, name='Pool3'))
    # FC1
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='FC1'))
    # FC2
    model.add(Dense(4096, activation='relu', name='FC2'))
    # FC3
    model.add(Dense(256, name='FC3'))
    model.add(BatchNormalization(name='Norm4'))

    if(remove_last_layer == False):
        # Dropout
        model.add(Dropout(0.1, name='Drop'))
        # FC4
        model.add(Dense(num_classes, activation='softmax', name='FC4'))

    return model

def create_model_inception(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the CNN model using inception blocks.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """

    inputs = Input(shape=(window_size, num_channels))
    block_1 = InceptionBlock(inputs, 1)
    block_2 = InceptionBlock(block_1, 2, 'flat')
    fc_1 = Dense(256, name='FC1')(block_2)
    
    # Model used for Identification
    if(remove_last_layer == False):
        fc_2 = Dense(num_classes, activation='softmax', name='FC2')(fc_1)
        model = Model(inputs=inputs, outputs=fc_2, name='Biometric_for_Identification')
        
    # Model used for Verification
    else:
        model = Model(inputs=inputs, outputs=fc_1, name='Biometric_for_Verification')

    return model

def create_model_SE(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the CNN model using squeeze & excitation blocks.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """

    inputs = Input(shape=(window_size, num_channels))
    block_1 = SEBlock(inputs)
    block_2 = SEBlock(block_1)
    block_3 = SEBlock(block_2)
    block_4 = SEBlock(block_3)
    block_5 = SEBlock(block_4, 'flat')
    fc_1 = Dense(256, name='FC1')(block_5)
    
    # Model used for Identification
    if(remove_last_layer == False):
        fc_2 = Dense(num_classes, activation='softmax', name='FC2')(fc_1)
        model = Model(inputs=inputs, outputs=fc_2, name='Biometric_for_Identification')
        
    # Model used for Verification
    else:
        model = Model(inputs=inputs, outputs=fc_1, name='Biometric_for_Verification')

    return model

def create_model_transformers(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the CNN model using transformers.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """

    inputs = Input(shape=(window_size, num_channels))
    x = MultiHeadAttention(num_heads=10, key_dim=num_channels)
    output_tensor = x(inputs, inputs)
    x = LayerNormalization() (output_tensor) # Add & Norm

    # x = Conv1D(96, (11), activation='relu') (x)
    # x = BatchNormalization() (x)
    # x = MaxPooling1D(strides=4) (x)

    x = Conv1D(96, (9), activation='relu') (x)
    x = BatchNormalization() (x)
    x = MaxPooling1D(strides=2) (x)

    x = Conv1D(128, (9), activation='relu') (x)
    x = BatchNormalization() (x)
    x = MaxPooling1D(strides=2) (x)

    x = Conv1D(256, (9), activation='relu') (x)
    x = BatchNormalization() (x)
    x = MaxPooling1D(strides=2) (x)

    x = Flatten() (x)
    x = Dense(4096)(x)
    x = Dense(4096)(x)
    x = Dense(256)(x)

    # Model used for Identification
    if(remove_last_layer == False):
        x = BatchNormalization()(x)
        x = Dropout(0.1) (x)
        x = Dense(num_classes, activation='softmax') (x)
        model = Model(inputs=inputs, outputs=x, name='Biometric_for_Identification')
        
    # Model used for Verification
    else:
        model = Model(inputs=inputs, outputs=x, name='Biometric_for_Verification')

    return model

def create_model_LSTM(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the CNN model using LSTM layers.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """

    model = None

    if(remove_last_layer == False):
        model = Sequential(name='Biometric_for_Identification')
    else:
        model = Sequential(name='Biometric_for_Verification')

    model.add(Input(shape=(window_size, num_channels)))
    model.add((LSTM(10, return_sequences=True)))
    model.add((LSTM(10, return_sequences=True)))
    model.add((LSTM(10, return_sequences=True)))
    model.add((LSTM(10, return_sequences=True)))
    model.add((LSTM(10, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(256))

    # Model used for Identification
    if(remove_last_layer == False):
        # model.add(BatchNormalization())
        # model.add(Dropout(0.1))
        model.add(Dense(num_classes, activation='softmax'))

    return model

def create_model_GRU(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the CNN model using GRU layers.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """

    model = None

    if(remove_last_layer == False):
        model = Sequential(name='Biometric_for_Identification')
    else:
        model = Sequential(name='Biometric_for_Verification')

    model.add(Input(shape=(window_size, num_channels)))
    model.add((GRU(10, return_sequences=True)))
    model.add((GRU(10, return_sequences=True)))
    model.add((GRU(10, return_sequences=True)))
    model.add((GRU(10, return_sequences=True)))
    model.add((GRU(10, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(256))

    # Model used for Identification
    if(remove_last_layer == False):
        # model.add(BatchNormalization())
        # model.add(Dropout(0.1))
        model.add(Dense(num_classes, activation='softmax'))

    return model

def create_model_mixed(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the CNN model that has multi-head attention, inception and sequeeze & excitation
    blocks mixed within.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """
    model = None

    if(remove_last_layer == False):
        model = Sequential(name='Biometric_for_Identification')
    else:
        model = Sequential(name='Biometric_for_Verification')

    model.add(Input(shape=(window_size, num_channels)))

 
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))

    model.add(Conv1D(96, (11), input_shape=(window_size, num_channels), activation='relu', name='Conv1'))
    model.add(BatchNormalization(name='Norm1'))
    model.add(MaxPooling1D(strides=4, name='Pool1'))
    
    model.add(Conv1D(128, (9), activation='relu', name='Conv2'))
    model.add(BatchNormalization(name='Norm2'))
    model.add(MaxPooling1D(strides=2, name='Pool2'))
    
    model.add(Conv1D(256, (9), activation='relu', name='Conv3')) 
    model.add(BatchNormalization(name='Norm3'))
    model.add(MaxPooling1D(strides=2, name='Pool3'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='FC1'))
    model.add(Dense(4096, activation='relu', name='FC2'))
    model.add(Dense(256, name='FC3'))
    model.add(BatchNormalization(name='Norm4'))

    if(remove_last_layer == False):
        model.add(Dropout(0.1, name='Drop'))
        model.add(Dense(num_classes, activation='softmax', name='FC4'))

    return model

def create_model_sun(window_size, num_channels, num_classes, remove_last_layer=False):
    """
    Creates and returns the CNN model used in Yingnan Sun et al. article, EEG-based user identification
    system using 1D-convolutional long short-term memory neural networks.

    Parameters:
        - window_size: sliding window size, used when composing the dataset;
        - num_channels: number of channels in an EEG signal;
        - num_classes: total number of classes (individuals).
    Optional Parameters:
        - remove_last_layer: if True, the model created won't have the fully connected block at the end with a
        softmax activation function.
    """
    inputs = Input(shape=(window_size, num_channels))

    x = (Conv1D(128, (2), padding='same', activation='relu', name='Layer_1')) (inputs)
    x = (Conv1D(256, (2), padding='same', activation='relu', name='Layer_2')) (x)
    x = (Conv1D(512, (2), padding='same', activation='relu', name='Layer_3')) (x)
    x = (Conv1D(1024, (2), padding='same', activation='relu', name='Layer_4')) (x)

    # n_ch = 64 * 16

    # Construct the LSTM inputs and LSTM cells
    # lstm_in = transpose(x, [1, 0, 2])  # reshape into (seq_len, batch, channels)
    # lstm_in = reshape(lstm_in, [-1, n_ch])  # Now (seq_len*N, n_channels)
    # # To cells
    # lstm_in = Dense(192) (lstm_in)  # or activation = tf.nn.relu
    # # Open up the tensor into a list of seq_len pieces
    # lstm_in = split(lstm_in, 160, 0)

    # Add LSTM layers
    lstm = [LSTMCell(192) for _ in range(2)]
    cell = StackedRNNCells(lstm)
    x = RNN(cell) (x)

    x = Dropout(0.5) (x)
    x = Dense(200, name='Layer_8') (x)
    x = Dense(200, name='Layer_9') (x)

    # Model used for Identification
    if(remove_last_layer == False):
        x = Dense(num_classes, activation='softmax') (x)
        model = Model(inputs=inputs, outputs=x, name='Biometric_for_Identification')
        
    # Model used for Verification
    else:
        model = Model(inputs=inputs, outputs=x, name='Biometric_for_Verification')

    return model

    # model.add(Flatten(name='Layer_5-1'))
    # model.add(Dense(192, name='Layer_5-2'))
    # model.add(Dropout(0.5, name='Layer_5-3'))

    # model.add(Reshape((-1, 192), name='Flatten_1'))
    # model.add(LSTM(192, return_sequences=True, name='Layer_6')) #activation='sigmoid'
    # model.add(LSTM(192, return_sequences=True, name='Layer_7')) #activation='sigmoid'

    # model.add(Flatten(name='Flatten_2'))
    # model.add(Dense(200, name='Layer_8')) # 192 units
    # model.add(Dense(200, name='Layer_9')) # 192 units

    # if(remove_last_layer == False):
    #     model.add(Dense(num_classes, activation='softmax', name='Layer_10'))

    # return model

### Redes Neurais e Aprendizagem em Profundidade###
# ResNet
import tensorflow.keras as keras

def create_model_resnet_1D(input_shape, nb_classes):
    n_feature_maps = 64

    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Inception
bottleneck_size = 32
nb_filters = 32
kernel_size = 41
depth = 6

def inception_module(input_tensor, stride=1, activation='linear'):

    if int(input_tensor.shape[-1]) > bottleneck_size:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                                padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                                strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                    padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

def shortcut_layer(input_tensor, out_tensor):
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                        padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def create_model_inception_1D(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = inception_module(x)

        if d % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

# FCN (Fully Convolutional Network)
def create_model_fcn(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model 

def create_model_resnet_1D_v2(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1 - CHANGED # OF FILTERS TO 96

    conv_x = keras.layers.Conv1D(filters=96, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=96, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=96, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=96, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2 - CHANGED # OF FILTERS TO 128

    conv_x = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=128, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3 - CHANGED # OF FILTERS OF TO 128

    conv_x = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=128, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

def create_model_resnet_1D_v1_lstm(input_shape, nb_classes):
    n_feature_maps = 64

    input_layer = keras.layers.Input(input_shape)

    # ADDING LSTM LAYERS
    lstm_layer = LSTM(128, return_sequences=True)(input_layer)
    lstm_layer = LSTM(128, return_sequences=True)(lstm_layer)
    lstm_layer = LSTM(128, return_sequences=True)(lstm_layer)
    lstm_layer = LSTM(128, return_sequences=True)(lstm_layer)
    lstm_layer = LSTM(128, return_sequences=True)(lstm_layer)

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(lstm_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

# def create_model_causal(window_size, num_channels, num_classes, remove_last_layer=False):
#     model = None

#     if(remove_last_layer == False):
#         model = Sequential(name='Biometric_for_Identification')
#     else:
#         model = Sequential(name='Biometric_for_Verification')

#     model.add(Input(shape=(window_size, num_channels)))

#     model.add(LSTM(128, return_sequences=True))
#     model.add(LSTM(128, return_sequences=True))
#     model.add(LSTM(128, return_sequences=True))
#     model.add(LSTM(128, return_sequences=True))
#     model.add(LSTM(128, return_sequences=True))

#     model.add(Conv1D(96, (11), input_shape=(window_size, num_channels), activation='relu', padding='causal', name='Conv1'))
#     model.add(BatchNormalization(name='Norm1'))
#     model.add(MaxPooling1D(strides=4, name='Pool1'))
    
#     model.add(Conv1D(128, (9), activation='relu', padding='causal', name='Conv2'))
#     model.add(BatchNormalization(name='Norm2'))
#     model.add(MaxPooling1D(strides=2, name='Pool2'))
    
#     model.add(Conv1D(256, (9), activation='relu', padding='causal', name='Conv3')) 
#     model.add(BatchNormalization(name='Norm3'))
#     model.add(MaxPooling1D(strides=2, name='Pool3'))

#     model.add(Flatten())
#     model.add(Dense(4096, activation='relu', name='FC1'))
#     model.add(Dense(4096, activation='relu', name='FC2'))
#     model.add(Dense(256, name='FC3'))
#     model.add(BatchNormalization(name='Norm4'))

#     if(remove_last_layer == False):
#         model.add(Dropout(0.1, name='Drop'))
#         model.add(Dense(num_classes, activation='softmax', name='FC4'))

#     return model

### incluindo dropout
def create_model_causal(window_size, num_channels, num_classes, remove_last_layer=False):
    model = None

    if(remove_last_layer == False):
        model = Sequential(name='Biometric_for_Identification')
    else:
        model = Sequential(name='Biometric_for_Verification')

    model.add(Input(shape=(window_size, num_channels)))

    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))

    model.add(Conv1D(512, (11), input_shape=(window_size, num_channels), activation='relu', padding='causal', name='Conv1'))
    model.add(BatchNormalization(name='Norm1'))
    model.add(MaxPooling1D(strides=4, name='Pool1'))
    # model.add(Dropout(0.2))

    model.add(Conv1D(128, (9), activation='relu', padding='causal', name='Conv2'))
    model.add(BatchNormalization(name='Norm2'))
    model.add(MaxPooling1D(strides=2, name='Pool2'))
    # model.add(Dropout(0.2))

    model.add(Conv1D(512, (9), activation='relu', padding='causal', name='Conv3')) 
    model.add(BatchNormalization(name='Norm3'))
    model.add(MaxPooling1D(strides=2, name='Pool3'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name='FC1',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1024, activation='relu', name='FC2',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1024, name='FC3',kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization(name='Norm4'))

    if(remove_last_layer == False):
        model.add(Dropout(0.2, name='Drop'))
        model.add(Dense(num_classes, activation='softmax', name='FC4'))

    return model
##fim



def create_cnn_based_on(window_size, num_channels, num_classes, remove_last_layer=False):
    model = Sequential(name='Regularized_CNN')

    model.add(Input(shape=(window_size, num_channels)))

    model.add(Conv1D(256, kernel_size=5, activation='relu', padding='causal', name='Conv1'))
    model.add(BatchNormalization(name='Norm1'))
    model.add(MaxPooling1D(strides=2, name='Pool1'))

    model.add(Conv1D(128, kernel_size=5, activation='relu', padding='causal', name='Conv2'))
    model.add(BatchNormalization(name='Norm2'))
    model.add(MaxPooling1D(strides=2, name='Pool2'))

    model.add(Conv1D(256, kernel_size=5, activation='relu', padding='causal', name='Conv3'))
    model.add(BatchNormalization(name='Norm3'))
    model.add(MaxPooling1D(strides=2, name='Pool3'))

    model.add(Flatten())

    model.add(Dense(128, activation='relu', name='FC1'))
    model.add(Dropout(0.3, name='Dropout_FC1'))
  
    model.add(Dense(num_classes, activation='relu', name='FC2'))
    model.add(Dropout(0.2, name='Dropout_FC2'))
    
    
    if not remove_last_layer:
        model.add(Dropout(0.2, name='Drop'))
        model.add(Dense(num_classes, activation='softmax', name='Output'))

    return model



def create_model_dilation(window_size, num_channels, num_classes, remove_last_layer=False):
    model = None

    if(remove_last_layer == False):
        model = Sequential(name='Biometric_for_Identification')
    else:
        model = Sequential(name='Biometric_for_Verification')

    model.add(Input(shape=(window_size, num_channels)))

    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))

    model.add(Conv1D(96, (11), input_shape=(window_size, num_channels), activation='relu', padding='causal', dilation_rate = 4, name='Conv1'))
    model.add(BatchNormalization(name='Norm1'))
    
    model.add(Conv1D(128, (9), activation='relu', padding='causal', dilation_rate = 2, name='Conv2'))
    model.add(BatchNormalization(name='Norm2'))
    
    model.add(Conv1D(256, (9), activation='relu', padding='causal', dilation_rate = 2, name='Conv3')) 
    model.add(BatchNormalization(name='Norm3'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='FC1'))
    model.add(Dense(4096, activation='relu', name='FC2'))
    model.add(Dense(256, name='FC3'))
    model.add(BatchNormalization(name='Norm4'))

    if(remove_last_layer == False):
        model.add(Dropout(0.1, name='Drop'))
        model.add(Dense(num_classes, activation='softmax', name='FC4'))

    return model
