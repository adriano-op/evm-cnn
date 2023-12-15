from sklearn.metrics.pairwise import euclidean_distances

import loader

# Data without Data Generators
train_content, test_content = loader.load_data(folder_path, train_tasks, test_tasks, 'csv', num_classes)

# Filtering the raw data
train_content = preprocessing.filter_data(train_content, band_pass_3, sample_frequency, filter_order, filter_type)
test_content = preprocessing.filter_data(test_content, band_pass_3, sample_frequency, filter_order, filter_type)

# Normalize the filtered data
train_content = preprocessing.normalize_data(train_content, 'sun')
test_content = preprocessing.normalize_data(test_content, 'sun')

# Getting the training, validation and testing data
training_generator = data_manipulation.DataGenerator(x_train_2_list, batch_size, window_size, offset,
    num_channels, num_classes, train_tasks, 'train', split_ratio, processed_data_path)
validation_generator = data_manipulation.DataGenerator(x_train_2_list, batch_size, window_size, offset,
    num_channels, num_classes, train_tasks, 'validation', split_ratio, processed_data_path)

######################################################################################

# Loading the data
train_content_2, test_content_2 = loader.load_data(folder_path, train_tasks, test_tasks, 'csv', num_classes)

# Filtering the raw data
test_content_2 = preprocessing.filter_data(test_content_2, band_pass_3, sample_frequency, filter_order, filter_type)

# Normalize the filtered data
test_content_2 = preprocessing.normalize_data(test_content_2, 'sun')

full_signal_size = test_content_2[0].shape[1]

# Getting the testing data
x_test_2, y_test_2 = data_manipulation.crop_data(test_content_2, test_tasks, num_classes, window_size,
                                    window_size)

for task in train_tasks:
    if(not os.path.exists(processed_data_path + 'processed_data/task'+str(task))):
        folder = Path(processed_data_path + 'processed_data/task'+str(task))
        folder.mkdir(parents=True)

        # Loading the raw data
        train_content_2, test_content_2 = loader.load_data(folder_path, [task], [], 'csv', num_classes)

        # Filtering the raw data
        train_content_2 = preprocessing.filter_data(train_content_2, band_pass_3, sample_frequency, filter_order, filter_type)

        # Normalize the filtered data
        train_content_2 = preprocessing.normalize_data(train_content_2, 'sun')

        list = []
        # for index in range(0, x_train_2.shape[0]):
        for index in range(0, len(train_content_2)):
            data = train_content_2[index]

            string = 'x_subject_' + str(index+1)
            savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv', data, fmt='%f', delimiter=';')
            print(processed_data_path + 'processed_data/task' + str(task) + '/' + string + '.csv was saved.')
            list.append(string+'.csv')
        
        savetxt(processed_data_path + 'processed_data/task' + str(task) + '/' + 'x_list.csv', [list], delimiter=',', fmt='%s')
        print(f'file names were saved to processed_data/task{task}/x_list.csv')

x_train_2_list = []

for task in train_tasks:
    x_train_2_list.append(loadtxt(processed_data_path + 'processed_data/task'+str(task)+'/x_list.csv', delimiter=',', dtype='str'))

x_train_2_list = [item for sublist in x_train_2_list for item in sublist]

# Defining the data generators
training_generator = data_manipulation.DataGenerator(x_train_2_list, batch_size, window_size, offset,
    full_signal_size, num_channels, num_classes, train_tasks, 'train', split_ratio, processed_data_path)
validation_generator = data_manipulation.DataGenerator(x_train_2_list, batch_size, window_size, offset,
    full_signal_size, num_channels, num_classes, train_tasks, 'validation', split_ratio, processed_data_path)

(x_train_2, y_train_2) = training_generator.return_all_data()
(x_val_2, y_val_2) = validation_generator.return_all_data()

print(f'x_train.shape = {x_train.shape}; x_train_2.shape = {x_train_2.shape}')
print(f'y_train.shape = {y_train.shape}; y_train_2.shape = {y_train_2.shape}')
print(f'x_val.shape = {x_val.shape}; x_val_2.shape = {x_val_2.shape}')
print(f'y_val.shape = {y_val.shape}; y_val_2.shape = {y_val_2.shape}')
print(f'x_test.shape = {x_test.shape}; x_test_2.shape = {x_test_2.shape}')
print(f'y_test.shape = {y_test.shape}; y_test_2.shape = {y_test_2.shape}')

i = 0
max = 0
while i < len(x_train):
    e_train_x = euclidean_distances(x_train[i], x_train_2[i])
    if(np.amax(e_train_x.diagonal()) > max):
        max = np.amax(e_train_x.diagonal())
    i += 1
print(f'e_train_x.diagonal() = {max}')

i = 0
max = 0
while i < len(x_val):
    e_val_x = euclidean_distances(x_val[i], x_val_2[i])
    if(np.amax(e_val_x.diagonal()) > max):
        max = np.amax(e_val_x.diagonal())
    i += 1
print(f'e_val_x.diagonal() = {max}')

i = 0
max = 0
while i < len(x_test):
    e_test_x = euclidean_distances(x_test[i], x_test_2[i])
    if(np.amax(e_test_x.diagonal()) > max):
        max = np.amax(e_test_x.diagonal())
    i += 1
print(f'e_test_x.diagonal() = {max}')

e_train_y = euclidean_distances(y_train, y_train_2)
e_val_y = euclidean_distances(y_val, y_val_2)
e_test_y = euclidean_distances(y_test, y_test_2)

print(f'e_train_y.diagonal() = {np.amax(e_train_y.diagonal())}\n')
print(f'e_val_y.diagonal() = {np.amax(e_val_y.diagonal())}\n')
print(f'e_test_y.diagonal() = {np.amax(e_test_y.diagonal())}\n')