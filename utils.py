import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
# from pyedflib import EdfReader

# def create_csv_database_from_edf(edf_folder_path, csv_folder_path, num_classes, channels = None):
#     """
#     Creates a database with CSV files from the original Physionet database, that contains EEG Signals stored
#     in EDF files.

#     Parameters:
#         - edf_folder_path: path of the folder in which the the EDF files are stored;
#         - csv_folder_path: path of the folder in which the the CSV files will be stored;
#         - num_classes: total number of classes (individuals).
    
#     Optional Parameters:
#         - channels: list of channel codes that will be read from the edf files. By default,
#         this function reads all channels. The list containing all channel codes is:
#         ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
#         'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
#         'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
#         'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
#         'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
#         'O1..', 'Oz..', 'O2..', 'Iz..']
#     """
#     if(os.path.exists(csv_folder_path) == False):
#         os.mkdir(csv_folder_path)

#     subject = 1
#     while(subject <= num_classes):
#         if(os.path.exists(csv_folder_path+'/S{:03d}'.format(subject)) == False):
#             os.mkdir(csv_folder_path+'/S{:03d}'.format(subject))

#         task = 1
#         while(task <= 14):
#             data = read_EDF(edf_folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(subject, subject, task), channels)
#             savetxt(csv_folder_path+'/S{:03d}/S{:03d}R{:02d}.csv'.format(subject, subject, task), data,
#                     fmt='%d', delimiter=',')
#             task += 1

#         subject += 1

def verbose_each_10_percent(count, data_amount, flag):
    """
    Auxiliar function for optional verbose on other functions. Returns the flag, possibly modified.

    Parameters:
        - count: current data index that was processed;
        - data_amount: length of the list of data;
        - flag: current state of the flag.
    """
    if count == data_amount and flag < 10:
        print('100%')
        flag = 10
    elif count >= data_amount * 0.9 and flag < 9:
        print('90%...',end='')
        flag = 9
    elif count >= data_amount * 0.8 and flag < 8:
        print('80%...',end='')
        flag = 8
    elif count >= data_amount * 0.7 and flag < 7:
        print('70%...',end='')
        flag = 7
    elif count >= data_amount * 0.6 and flag < 6:
        print('60%...',end='')
        flag = 6
    elif count >= data_amount * 0.5 and flag < 5:
        print('50%...',end='')
        flag = 5
    elif count >= data_amount * 0.4 and flag < 4:
        print('40%...',end='')
        flag = 4
    elif count >= data_amount * 0.3 and flag < 3:
        print('30%...',end='')
        flag = 3
    elif count >= data_amount * 0.2 and flag < 2:
        print('20%...',end='')
        flag = 2
    elif count >= data_amount * 0.1 and flag < 1:
        print('10%...',end='')
        flag = 1
    
    return flag
    
def one_hot_encoding_to_classes(y_data):
    """
    Takes a 2D numpy array that contains one-hot encoded labels and returns a 1D numpy array that contains
    the classes.

    Parameters:
        - y_data: 2D numpy array in the format (number of samples, number of classes).
    """

    i = 0
    j = 0
    num_samples = y_data.shape[0]
    arr = np.zeros(shape=(num_samples, 1))

    while i < num_samples:
        while y_data[i, j] != 1:
            j += 1
        arr[i] = j+1
        i += 1
    
    return arr

def n_samples_with_sliding_window(start, end, window_size, offset):
    """
    Returns the number of samples in a signal, generated after applying a sliding window.

    Parameters:
        - start: starting position of the signal;
        - end: ending position of the signal;
        - window_size: size of the sliding window;
        - offset: amount of samples the window will slide in each iteration.
    """
    n_samples = 0
    i = start

    if(offset == 0):
        print('ERROR: An offset equal to 0 would result in "infinite" equal windows.')
        return 0

    if(i < end):
        n_samples = 1
        i += window_size

    while(i < end):
        n_samples += 1
        i += offset

    return n_samples

def calc_metrics(feature1, label1, feature2, label2, plot_det=True, path=None):
    """
    Calculates Decidability, Equal Error Rate (EER) and returns them, as well as the respective thresholds.

    Parameters:
        - feature1: one of the feature vectors;
        - label1: labels of the feature1 vector;
        - feature2: one of the feature vectors;
        - label2: labels of the feature2 vector.
    
    Optional Parameters:
        - plot_det: if set to True, plots the Detection Error Trade-Off (DET) graph. True by default;
        - path: file path that will store the Detection Error Trade-Off (DET) graph in a png file. No file path 
        is selected by default.
    """

    resolu = 5000

    feature1 = feature1.T
    xmax = np.amax(feature1,axis=0)
    xmin = np.amin(feature1,axis=0)
    x = feature1
    feature1 = (x - xmin)/(xmax - xmin)
    feature1 = feature1.T

    feature2 = feature2.T
    xmax = np.amax(feature2, axis=0)
    xmin = np.amin(feature2, axis=0)
    x = feature2
    feature2 = (x - xmin) / (xmax - xmin)
    feature2 = feature2.T

    # All against all euclidean distance
    dist = euclidean_distances(feature1, feature2)

    # Getting the smaller dimensions
    smaller1 = len(label1)
    if(len(feature1) < len(label1)):
        smaller1 = len(feature1)
    
    smaller2 = len(label2)
    if(len(feature2) < len(label2)):
        smaller2 = len(feature2)

    # Separating distances from genuine pairs and impostor pairs
    same_list = []
    dif_list = []
    for row in range(smaller1):
        for col in range(row+1, smaller2):
            if (label1[row] == label2[col]):
                same_list.append(dist[row, col])
            else:
                dif_list.append(dist[row, col])

    same = np.array(same_list)
    dif = np.array(dif_list)

    # Mean and standard deviation of both vectors
    mean_same = np.mean(same)
    mean_dif = np.mean(dif)
    std_same = np.std(same)
    std_dif = np.std(dif)

    # Decidability
    d = abs(mean_dif - mean_same) / np.sqrt(0.5 * (std_same ** 2 + std_dif ** 2))

    dmin = np.amin(same)
    dmax = np.amax(dif)

    # Calculate False Match Rate and False NonMatch Rate for different thresholds
    FMR = np.zeros(resolu)
    FNMR = np.zeros(resolu)
    t = np.linspace(dmin, dmax, resolu)

    for t_val in range(resolu):
        fm = np.sum(dif <= t[t_val])
        FMR[t_val] = fm / len(dif)

    for t_val in range(resolu):
        fnm = np.sum(same > t[t_val])
        FNMR[t_val] = fnm / len(same)

    # DET graph (FMR x FNMR)
    plt.plot(FMR, FNMR, color='darkorange', label='DET curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Match Rate')
    plt.ylabel('False NonMatch Rate')
    plt.title('Detection Error Trade-Off')
    plt.legend(loc="lower right")

    # If plot_det = True, plots FMR x FNMR
    if plot_det == True:
        plt.show()

    # If path != None, saves FMR x FNMR to a file
    if path != None:
        plt.savefig(path + r'EER.png', format='png')

    # Equal Error Rate (EER)
    abs_diffs = np.abs(FMR - FNMR)
    min_index = np.argmin(abs_diffs)
    eer = (FMR[min_index] + FNMR[min_index])/2
    thresholds = t[min_index]

    return d, eer, thresholds

class Logger(object):
    """
    Direct all output from terminal to an output file.
    """

    def __init__(self, output_file: str):
        """
        Constructor.
        :param output_file: the file which the console log will be written.
        """
        self.terminal = sys.stdout
        self.output_file = output_file
        self.log = None

    def write(self, message: str):
        """
        Append a message from the console log to the file.
        :param message:
        :return:
        """
        with open(self.output_file, 'a', encoding='utf-8') as self.log:
            self.log.write(message)
        self.terminal.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass