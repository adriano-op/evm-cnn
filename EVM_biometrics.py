import sys
import logging
import argparse
import os
import time
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
 
import matplotlib.pyplot as plt

import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from numpy import savetxt, loadtxt
from numpy import interp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
 

# Parameters used in functions.load_data()
folder_path = '/media/work/adrianoandrade/redesNeurais/Dataset_CSV_V_2/'
processed_data_path = '/media/work/adrianoandrade/redesNeurais/'
sys.path.append(processed_data_path)

import models
import data_manipulation
import preprocessing
import utils
import loader

# Logger setup
def setup_logging(training_epochs):
    log_filename = os.path.join(processed_data_path, 'results', f'log_script_{training_epochs}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

def plot_roc_media_multiclass(x_test, y_test,y_train,model,nome_modelo):

    # Fazer previsões no conjunto de validação
    y_pred = model.predict(x_test)

    # Binarizar as classes verdadeiras
    y_test_binarized = label_binarize(y_test, classes=np.arange(y_train.shape[1]))

    # Calcular as curvas ROC para cada classe
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_train.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    from numpy import interp

    # Calcular a média das curvas ROC para todas as classes
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i in range(y_train.shape[1]):
        tprs.append(interp(mean_fpr, fpr[i], tpr[i]))
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    

    

    # Plotar a curva ROC média
    plt.figure(figsize=(15, 15), dpi=600)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.xlabel('Taxa de Falso Positivo', fontsize=15)
    plt.ylabel('Taxa de Verdadeiro Positivo', fontsize=15)
    plt.title('Média da Curva ROC', fontsize=15)
    plt.legend()

    file_path = os.path.join(processed_data_path+'/teste_17_07_2024/plot_images', 'plot_roc_media_multiclas_' + nome_modelo + '.png')
    # file_path = os.path.join(processed_data_path+'/plot_images', 'plot_roc_media_multiclas_' + nome_modelo + '.png')
    # Salve a imagem
    plt.savefig(file_path, format='png')


    plt.show()

    return mean_auc

def plot_roc_curves(processed_data_path,model, X_test, y_test,nome_modelo):
    print('plot_roc_curves\n')
    """
    Plota todas as curvas ROC para cada classe em um modelo CNN multiclasse.

    Argumentos:
        model: O modelo CNN multiclasse treinado.
        X_test: Conjunto de dados de teste (features).
        y_test: Rótulos de classe verdadeiros do conjunto de teste.
    """

    # Obter scores de predição do modelo
    y_pred = model.predict(X_test)
    plt.figure(figsize=(20, 15), dpi=600)

    # Calcular ROC para cada classe
    num_classes = y_pred.shape[1]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)

        # Plotar curva ROC
        plt.plot(fpr, tpr, label=f'Classe {i} (AUC = {roc_auc:.3f})')

    # Adicionar legendas e título
    plt.xlabel('FPR (Taxa de Falso Positivo)')
    plt.ylabel('TPR (Taxa de Verdadeiro Positivo)')
    plt.title('Curvas ROC - Modelo CNN Multiclasse')
    plt.legend()

     # Caminho para salvar a imagem
    file_path = os.path.join(processed_data_path+'/teste_17_07_2024/plot_images', 'plot_roc_curves_' + nome_modelo + '.png')
    # file_path = os.path.join(processed_data_path+'/plot_images', 'plot_roc_curves_' + nome_modelo + '.png')

    # Salvar a imagem
    plt.savefig(file_path, format='png')

    # Mostrar o gráfico
    plt.show()



 

def plot_multiclass_roc(x_test, y_test, y_train, num_classes, processed_data_path, model, nome_modelo):
    print('plot_multiclass_roc \n')

    # Binarizar as classes de teste com base no número de classes de treinamento
    y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))

    # Inicializar dicionários para armazenar FPR, TPR e AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(x_test)

    # Calcular ROC e AUC para cada classe
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Agregar todos os pontos de ROC para calcular a média macro
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plotar todas as curvas ROC
    plt.figure(figsize=(15, 20), dpi=600)  # Aumentar o tamanho e a resolução
    colors = [
        "aqua", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "red", "silver", "teal", "yellow",
        "aliceblue", "antiquewhite", "beige", "bisque", "blanchedalmond", "blueviolet", "burlywood", "cadetblue", "chartreuse", "chocolate",
        "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen", "darkkhaki",
        "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
        "darkturquoise", "deeppink", "deepskyblue", "dimgray", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "gainsboro", "ghostwhite",
        "gold", "goldenrod", "greenyellow", "honeydew", "hotpink", "indianred", "khaki", "lavender", "lawngreen", "lemonchiffon", "lightblue",
        "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgreen", "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray",
        "lightsteelblue", "limegreen", "linen", "magenta", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumseagreen", "mediumslateblue",
        "mediumspringgreen", "mediumturquoise", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "oldlace", "orangered",
        "palegoldenrod", "palegreen", "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue",
        "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seashell", "skyblue", "slateblue", "snow", "springgreen", "steelblue",
        "tan", "thistle", "tomato", "turquoise", "wheat", "whitesmoke", "yellowgreen"
    ]
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--'] * (num_classes // 10 + 1)

    plt.plot(fpr["macro"], tpr["macro"], color='blue', linestyle='--',
             label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})')
    for i, color, linestyle in zip(range(num_classes), colors, linestyles):
        plt.plot(fpr[i], tpr[i], color=color, linestyle=linestyle, lw=2,
                 label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

    # Linha de referência para uma ROC de azar
    plt.plot([0, 1], [0, 1], 'k--')

    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title('Curva ROC Multiclasse', fontsize=18)
    plt.legend(loc="lower right")

    # Caminho para salvar a imagem
   
    file_path = os.path.join(processed_data_path+'/teste_17_07_2024/plot_images', 'plot_multiclass_roc_' + nome_modelo + '.png')

    # Salvar a imagem
    plt.savefig(file_path, format='png', bbox_inches='tight')

    # Mostrar o gráfico
    plt.show()


# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datagen', action='store_true', help='the model will use Data Generators to crop data on the fly')
parser.add_argument('--nofit', action='store_true', help='model.fit will not be executed. The weights will be gathered from the file' +
                    ' \'model_weights.keras\', that is generated if you have ran the model in Identification mode' +
                    ' at least once')
parser.add_argument('--noimode', action='store_true', help='the model won\'t run in Identification Mode')
parser.add_argument('--novmode', action='store_true', help='the model won\'t run in Verification Mode')
#parser.add_argument('-train', nargs="+", type=int, required=True, help='list of tasks used for training and validation. All specified tasks need to be higher than\n' +
#                    ' 0 and lower than 15. This is a REQUIRED flag')
#parser.add_argument('-test', nargs="+", type=int, required=True, help='list of tasks used for testing. All specified tasks need to be higher than 0 and lower than\n' +
#                    ' 15. This is a REQUIRED flag')
parser.add_argument('--training_epochs', type=int, required=True, help='Total number of training epochs')

args = parser.parse_args()

# Setup logging
setup_logging(args.training_epochs)
logging.info('Logger is set up')

# Seeds
random.seed(1051)
np.random.seed(1051)
tf.random.set_seed(1051)

# Hyperparameters
batch_size = 100                # Batch Size
training_epochs = args.training_epochs  # Total number of training epochs
initial_learning_rate = 0.01    # Initial learning rate

# Parameters used in functions.load_data()
num_classes = 56               # Total number of classes (individuals)

# Parameters used in functions.filter_data()
# band_pass_1 = [1, 50]           # First filter option, 1~50Hz
# band_pass_2 = [10, 30]          # Second filter option, 10~30Hz
# band_pass_3 = [30, 50]          # Third filter option, 30~50Hz
band_pass_4 = [0.01, 29]        # Four filter option,  00~30Hz

sample_frequency = 60           # Frequency of the sampling
filter_order = 12               # Order of the filter
filter_type = 'filtfilt'        # Type of the filter used: 'sosfilt' or 'filtfilt'

# Parameters used in functions.normalize_data()
normalize_type = 'each_channel' # Type of the normalization that will be applied: 'each_channel' or 'all_channels'

# Parameters used in functions.crop_data()
window_size = 300               # Sliding window size, used when composing the dataset
offset = 30                     # Sliding window offset (deslocation), used when composing the dataset
split_ratio = 0.9               # 90% for training | 10% for validation

# Other Parameters
num_channels = 1                # Number of channels in an EEG signal

#train_tasks = args.train
#test_tasks = args.test

train_tasks = [1,2]
test_tasks = []

for task in train_tasks:
    if(task <= 0 or task >= 15):
        logging.error('All training/validation and testing tasks need to be higher than 0 and lower than 15.\n')
        sys.exit()

for task in test_tasks:
    if(task <= 0 or task >= 15):
        logging.error('All training/validation and testing tasks need to be higher than 0 and lower than 15.\n')
        sys.exit()

# Defining the optimizer and the learning rate scheduler
opt = SGD(learning_rate=initial_learning_rate, momentum=0.9)
lr_scheduler = LearningRateScheduler(models.scheduler, verbose=0)
saver = models.SaveAtEpochEnd(5, 'model_weights')
model = None

logging.info(f'args.datagen: {args.datagen}')

if not args.datagen:
    logging.info('Not Using Data Generators')
    print('\n\n Not Using Data Generators \n')
    
    train_content, test_content = loader.load_data(folder_path, train_tasks, test_tasks, 'csv', num_classes, 1)
    train_content = preprocessing.filter_data(train_content, band_pass_4, sample_frequency, filter_order, filter_type, 1)
    
    x_train, y_train, x_test, y_test = data_manipulation.crop_data(train_content, train_tasks, num_classes, window_size, offset, split_ratio)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=31)
    
    scaler = MinMaxScaler()
    x_train_normalized = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1]))
    x_val_normalized = scaler.transform(x_val.reshape(-1, x_val.shape[-1]))
    x_test_normalized = scaler.transform(x_test.reshape(-1, x_test.shape[-1]))
    
    x_train = x_train_normalized.reshape(x_train.shape)
    x_val = x_val_normalized.reshape(x_val.shape)
    x_test = x_test_normalized.reshape(x_test.shape)

    logging.info(f' x_train, x_val, y_train, y_val, x_test.shape, y_test.shape: {x_train.shape}, {x_val.shape}, {y_train.shape}, {y_val.shape}, {x_test.shape}, {y_test.shape}')
   # print(f' x_train, x_val, y_train, y_val, x_test.shape, y_test.shape: {x_train.shape}, {x_val.shape}, {y_train.shape}, {y_val.shape}, {x_test.shape}, {y_test.shape}')

    if not args.nofit:
        print('\n\n Not Using nofit \n')
        model = models.create_model_13_porcento(window_size, num_channels, num_classes)
        model.summary()
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

        fit_begin = time.time()

        results = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=training_epochs,
            callbacks=[lr_scheduler],
            validation_data=(x_val, y_val)
        )

        fit_end = time.time()
        logging.info(f'Modelo: create_model_13_porcento')
        logging.info(f'Training time in seconds: {fit_end - fit_begin}')
        logging.info(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
        logging.info(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

        plt.subplot(211)
        plt.plot(results.history['accuracy'])
        plt.plot(results.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])

        plt.subplot(212)
        plt.plot(results.history['loss'])
        plt.plot(results.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.tight_layout()
        plt.savefig(r'accuracy-loss.png', format='png')
        plt.show()

        max_loss = np.max(results.history['loss'])
        min_loss = np.min(results.history['loss'])
        logging.info(f"Maximum Loss : {max_loss:.4f}")
        logging.info(f"Minimum Loss : {min_loss:.4f}")
        logging.info(f"Loss difference : {(max_loss - min_loss):.4f}\n")

        model.save('model_weights.keras')
        logging.info('Model was saved to model_weights.keras.\n')
        print('Model was saved to model_weights.keras.\n')

    if not args.noimode:
        print('noimode : \n')
        logging.info('noimode:')
        if model is None:
            model = models.create_model_13_porcento(window_size, num_channels, num_classes)
            model.summary()
            model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.load_weights('model_weights.keras', by_name=True)

        logging.info('\nEvaluating on training set...')
        loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
        logging.info(f'loss={loss:.4f}, accuracy: {accuracy*100:.4f}%\n')

        logging.info('Evaluating on validation set...')
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
        logging.info(f'loss={loss:.4f}, accuracy: {accuracy*100:.4f}%\n')

        logging.info('Evaluating on testing set...')
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        logging.info(f'loss={loss:.4f}, accuracy: {accuracy*100:.4f}%\n')

    if  not args.novmode:
        # Removing the last layers of the model and getting the features array
        print('\n model_for_verification :\n')
        model_for_verification = models.create_model_13_porcento(window_size, num_channels, num_classes, True)
        model_for_verification.summary()
        model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model_for_verification.load_weights('model_weights.keras', by_name=True)

        x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

        # Calculating EER and Decidability
        # y_test_classes = utils.one_hot_encoding_to_classes(y_test)
        # d, eer, thresholds = utils.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
        # logging.info(f'EER: {eer*100.0} %')
        # logging.info(f'Decidability: {d}')
        # print(f'EER: {eer*100.0} %')
        # print(f'Decidability: {d}')

	# calculate the genuine curve and impostor curve
        print('\n\n\n\n')
        print('XXXXXXXXXXXXXXXXXXXXXXXX  Dados Crus: ')
        logging.info('XXXXXXXXXXXXXXXXXXXXXXXX  Dados Crus :')
        x_test1 = x_test.reshape((-1, window_size))
        d, eer =utils.biometrics(x_test1, y_test )
        
        print(f'EER cru: {eer*100.0} %')
        print(f'Decidability cru: {d}')
        logging.info(f'EER cru: {eer*100.0} %')
        logging.info(f'Decidability cru: {d}')

        print('\n\n\n\n')
        print('XXXXXXXXXXXXXXXXXXXXXXXX  Dados trabalhados :')
        logging.info('XXXXXXXXXXXXXXXXXXXXXXXX  Dados tratados :')
        d, eer =utils.biometrics(x_pred, y_test )
        print(f'EER tratado: {eer*100.0} %')
        print(f'Decidability tratado: {d}')
        logging.info(f'EER tratado: {eer*100.0} %')
        logging.info(f'Decidability tratado: {d}')
        
         
       
        mean_auc = plot_roc_media_multiclass(x_test, y_test,y_train,model,'create_model_13_porcento')
        print('mean_auc',mean_auc)
     
        plot_roc_curves(processed_data_path,model, x_test, y_test,'create_model_13_porcento')

        plot_multiclass_roc(x_test,y_test, y_train, num_classes, processed_data_path,model, 'create_model_13_porcento')
       


else:
    print('aqui,Using Data Generators')
    
    train_content, test_content = loader.load_data(folder_path, train_tasks, test_tasks, 'csv', num_classes, 1)
    train_content = preprocessing.filter_data(train_content, band_pass_4, sample_frequency, filter_order, filter_type, 1)
    
    x_train, y_train, x_test, y_test = data_manipulation.crop_data(train_content, train_tasks, num_classes, window_size, offset, split_ratio)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=31)
    
    scaler = MinMaxScaler()
    x_train_normalized = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1]))
    x_val_normalized = scaler.transform(x_val.reshape(-1, x_val.shape[-1]))
    x_test_normalized = scaler.transform(x_test.reshape(-1, x_test.shape[-1]))
    
    x_train = x_train_normalized.reshape(x_train.shape)
    x_val = x_val_normalized.reshape(x_val.shape)
    x_test = x_test_normalized.reshape(x_test.shape)

    logging.info(f' x_train, x_val, y_train, y_val, x_test.shape, y_test.shape: {x_train.shape}, {x_val.shape}, {y_train.shape}, {y_val.shape}, {x_test.shape}, {y_test.shape}')
   # print(f' x_train, x_val, y_train, y_val, x_test.shape, y_test.shape: {x_train.shape}, {x_val.shape}, {y_train.shape}, {y_val.shape}, {x_test.shape}, {y_test.shape}')

    # Processing train/validation data
    for task in train_tasks:

        if(not os.path.exists(processed_data_path + 'processed_data/task'+str(task))):
            folder = Path(processed_data_path + 'processed_data/task'+str(task))
            folder.mkdir(parents=True)

            # Loading the raw data
            train_content, test_content = loader.load_data(folder_path, [task], [], 'csv', num_classes)

            # Filtering the raw data
            train_content = preprocessing.filter_data(train_content, band_pass_3, sample_frequency, filter_order, filter_type)

            # Normalize the filtered data
            train_content = preprocessing.normalize_data(train_content, 'sun')

            list = []
            for index in range(0, len(train_content)):
                data = train_content[index]
                string = 'x_subject_' + str(index+1)
                savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv', data, fmt='%f', delimiter=';')
                print(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv was saved.')
                list.append(string+'.csv')
                
            savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + 'x_list.csv', [list], delimiter=',', fmt='%s')
            print(f'file names were saved to processed_data/task{task}/x_list.csv')

    # Getting the file names that contains the preprocessed data
    x_train_list = []

    for task in train_tasks:
        x_train_list.append(loadtxt(processed_data_path + 'processed_data/task'+str(task)+'/x_list.csv', delimiter=',', dtype='str'))

    x_train_list = [item for sublist in x_train_list for item in sublist]

    # Defining the data generators
    training_generator = data_manipulation.DataGenerator(x_train_list, batch_size, window_size, offset,
        num_channels, num_classes, train_tasks, 'train', split_ratio, processed_data_path, True)
    validation_generator = data_manipulation.DataGenerator(x_train_list, batch_size, window_size, offset,
        num_channels, num_classes, train_tasks, 'validation', split_ratio, processed_data_path, True)
    
    if not args.nofit:
        logging.info('nofit : \n')
        print('nofit :\n')
        train_generator, val_generator, train_num_steps, val_num_steps = loader.load_generators(folder_path, train_tasks, window_size, offset, num_classes, sample_frequency, batch_size, filter_order, filter_type, 1)
        model = models.create_model_13_porcento(window_size, num_channels, num_classes)
        model.summary()
        model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

        fit_begin = time.time()

        results = model.fit(
            train_generator,
            steps_per_epoch=train_num_steps,
            epochs=training_epochs,
            callbacks=[lr_scheduler],
            validation_data=val_generator,
            validation_steps=val_num_steps
        )

        fit_end = time.time()
        logging.info(f'Training time in seconds: {fit_end - fit_begin}')
        logging.info(f'Training time in minutes: {(fit_end - fit_begin)/60.0}')
        logging.info(f'Training time in hours: {(fit_end - fit_begin)/3600.0}\n')

        plt.subplot(211)
        plt.plot(results.history['accuracy'])
        plt.plot(results.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])

        plt.subplot(212)
        plt.plot(results.history['loss'])
        plt.plot(results.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'])
        plt.tight_layout()
        plt.savefig(r'accuracy-loss.png', format='png')
        plt.show()

        max_loss = np.max(results.history['loss'])
        min_loss = np.min(results.history['loss'])
        logging.info(f"Maximum Loss : {max_loss:.4f}")
        logging.info(f"Minimum Loss : {min_loss:.4f}")
        logging.info(f"Loss difference : {(max_loss - min_loss):.4f}\n")

        model.save('model_weights.keras')
        logging.info('Model was saved to model_weights.keras.\n')

    if not args.noimode:
        print('noimode :\n')
        logging.info('noimode :\n')
        if model is None:
            model = models.create_model_13_porcento(window_size, num_channels, num_classes)
            model.summary()
            model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
            model.load_weights('model_weights.keras', by_name=True)

        logging.info('\nEvaluating on training set...')
        loss, accuracy = model.evaluate(train_generator, steps=train_num_steps, verbose=0)
        logging.info(f'loss={loss:.4f}, accuracy: {accuracy*100:.4f}%\n')

        logging.info('Evaluating on validation set...')
        loss, accuracy = model.evaluate(val_generator, steps=val_num_steps, verbose=0)
        logging.info(f'loss={loss:.4f}, accuracy: {accuracy*100:.4f}%\n')
        test_end = time.time()
        print(f'Evaluating on testing set time in miliseconds: {(test_end - test_begin) * 1000.0}')
        print(f'Evaluating on testing set time in seconds: {test_end - test_begin}')
        print(f'Evaluating on testing set time in minutes: {(test_end - test_begin)/60.0}\n')
    
    # Running the model in Verification Mode
    if(not args.novmode):
        print('model_for_verification DATA_GEN : \n')
        logging.info('model_for_verification DATA_GEN : \n')
        # Removing the last layers of the model and getting the features array
        model_for_verification = models.create_model_causal(window_size, num_channels, num_classes, True) ##
        model_for_verification.summary()
        model_for_verification.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model_for_verification.load_weights('model_weights.keras', by_name=True)

        x_pred = model_for_verification.predict(x_test, batch_size = batch_size)

        # Calculating EER and Decidability
        y_test_classes = utils.one_hot_encoding_to_classes(y_test)
        d, eer, thresholds = utils.calc_metrics(x_pred, y_test_classes, x_pred, y_test_classes)
        logging.info(f'EER: {eer*100.0} %')
        logging.info(f'Decidability: {d}')

	# calculate the genuine curve and impostor curve
        
        print('\n\n\n\n')
        print('XXXXXXXXXXXXXXXXXXXXXXXX  Dados Crus  XXXXXXXXXXXXXXXXXXXXXXXX')
        x_test1 = x_test.reshape((-1, window_size))
        d, eer =utils.biometrics(x_test1, y_test )
        print(f'EER cru: {eer*100.0} %')
        print(f'Decidability cru: {d}')
        logging.info(f'EER cru: {eer*100.0} %')
        logging.info(f'Decidability cru: {d}')

        print('\n\n\n\n')
        print('XXXXXXXXXXXXXXXXXXXXXXXX  Dados trabalhados XXXXXXXXXXXXXXXXXXXXXXXX')
        d, eer =utils.biometrics(x_pred, y_test )
        print(f'EER tratado: {eer*100.0} %')
        print(f'Decidability tratado: {d}')
        logging.info(f'EER tratado: {eer*100.0} %')
        logging.info(f'Decidability tratado: {d}')

        mean_auc = plot_roc_media_multiclass(x_test, y_test,y_train,model,'create_model_13_porcento')
        print('mean_auc',mean_auc)
     
        plot_roc_curves(processed_data_path,model, x_test, y_test,'create_model_13_porcento')

        plot_multiclass_roc(x_test,y_test, y_train, num_classes, processed_data_path,model, 'create_model_13_porcento')





