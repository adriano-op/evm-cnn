# from pyedflib import EdfReader
from numpy import savetxt, loadtxt

# def read_EDF(path, channels = None):
#     """
#     Reads data from an EDF file and returns it in a numpy array format.

#     Parameters:
#         - path: path of the file that will be read.
    
#     Optional Parameters:
#         - channels: list of channel codes that will be read. By default, this function reads all channels.
#         The list containing all channel codes is: ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
#         'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.',
#         'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..',
#         'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..',
#         'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
#         'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
#     """
    
#     file_folder = os.path.dirname(os.path.abspath(__file__))
#     new_path = os.path.join(file_folder, path)
#     reader = EdfReader(new_path)

#     if channels:
#         signals = []
#         signal_labels = reader.getSignalLabels()
#         for c in channels:
#             index = signal_labels.index(c)
#             signals.append(reader.readSignal(index))
#         signals = np.array(signals)
#     else:
#         n = reader.signals_in_file
#         signals = np.zeros((n, reader.getNSamples()[0]))
#         for i in np.arange(n):
#             signals[i, :] = reader.readSignal(i)

#     reader._close()
#     del reader
#     return signals

def load_data(folder_path, train_tasks, test_tasks, file_type, num_classes, channels = None, verbose = 0):
    """
    Loads and returns lists containing raw signals used for training (train_content) and testing (test_content).

    The return of this function is in the format: train_content, test_content.

    Parameters:
        - folder_path: path of the folder in which the the EDF files are stored.
        E.g. if this python script is in the same folder as the sub-folder used to store the EDF files, and this
        sub-folder is called "Dataset", then this parameter should be: './Dataset/';
        - train_tasks: list that contains the numbers of the experimental runs that will be used to create train
        and validation data;
        - test_tasks: list that contains the numbers of the experimental runs that will be used to create testing
        data;
        - file_type: extension of the files that contains the EEG signals. Valid extensions are 'edf' and 'csv';
        - num_classes: total number of classes (individuals).
    
    Optional Parameters:
        - channels: this parameter is only used when file_type is 'edf'. List of channel codes that will be read.
        By default, this function reads all channels. The list containing all channel codes is: ['Fc5.', 'Fc3.',
        'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.',
        'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.',
        'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..',
        'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..',
        'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
        - verbose: if set to 1, prints what type of data (training/validation or testing) is currently being
        loaded. Default value is 0.
    """

    # Processing x_train, y_train, x_val and y_val
    if(verbose):
        print('Training and Validation data are being loaded...')

    train_content = list()

    for train_task in train_tasks:
        if(verbose):
            print(f'* Using task {train_task}:')

        for i in range(1, num_classes + 1):
            if(verbose):
                print(f'  > Loading data from subject {i}.')

            # if(file_type == 'edf'):
                # train_content.append(read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, train_task), channels))
            if(file_type == 'csv'):
                # train_content.append(loadtxt(folder_path+'S{:03d}/S{:03d}R{:02d}.csv'.format(i, i, train_task), delimiter=','))
                train_content.append(loadtxt(folder_path+'S{:03d}/S{:03d}R{:02d}.csv'.format(i, i, train_task), delimiter=',',ndmin=2))
            else:
                print('ERROR: Invalid file_type parameter. Data will not be loaded.')

    # Processing x_test and y_test
    if(verbose):
        print('\nTesting data are being loaded...')

    test_content = list()

    for test_task in test_tasks:
        if(verbose):
            print(f'* Using task {test_task}:')

        for i in range(1, num_classes + 1):
            if(verbose):
                print(f'  > Loading data from subject {i}.')

            # if(file_type == 'edf'):
                # test_content.append(read_EDF(folder_path+'S{:03d}/S{:03d}R{:02d}.edf'.format(i, i, test_task), channels))
            if(file_type == 'csv'):
                # test_content.append(loadtxt(folder_path+'S{:03d}/S{:03d}R{:02d}.csv'.format(i, i, test_task), delimiter=','))
                test_content.append(loadtxt(folder_path+'S{:03d}/S{:03d}R{:02d}.csv'.format(i, i, test_task), delimiter=',',ndmin=2))
            else:
                print('ERROR: Invalid file_type parameter. Data will not be loaded.')

    return train_content, test_content