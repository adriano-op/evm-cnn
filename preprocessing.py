import utils

import numpy as np
from scipy.signal import butter, sosfilt, firwin, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, filter_order, filter_type):
    """
    Band-pass filters a signal and returns it.

    Parameters:
        - signal: signal that will be band-pass filtered;
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - fs: frequency of the signal;
        - filter_order: order of the filter;
        - filter_type: how the signal will be filtered:
            * 'sosfilt': using the sosfilt() function from the scipy library;
            * 'filtfilt': using the firwin() and filtfilt() functions from the scipy library.
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # print('bandpass_filter',filter_order+1,[low,high])
    if(filter_type == 'sosfilt'):
        sos = butter(filter_order, [low, high], btype='band', output='sos')
        y = sosfilt(sos, signal)
    elif(filter_type == 'filtfilt'):
        fir_coeff = firwin(filter_order+1,[low,high], pass_zero=False)
        y = filtfilt(fir_coeff, 1.0, signal)

    return y

def pre_processing(content, lowcut, highcut, frequency, filter_order, filter_type):
    """
    Pre-processess each channel of an EEG signal using band-pass filters.

    Parameters:
        - signal: signal that will be band-pass filtered;
        - lowcut: lowcut of the filter;
        - highcut: highcut of the filter;
        - fs: frequency of the signal;
        - filter_order: order of the filter;
        - filter_type: type of the filter used:
            * 'sosfilt': using the sosfilt() function from the scipy library.
            * 'filtfilt': using the firwin() and filtfilt() functions from the scipy library.
    """

    channels = content.shape[0]
    c = 0

    if(filter_type != 'sosfilt' and filter_type != 'filtfilt'):
        print('ERROR: Invalid filter_type parameter. Signal will not be filtered.')
        return content

    while c < channels:
        signal = content[c, :]
        content[c] = bandpass_filter(signal, lowcut, highcut, frequency, filter_order, filter_type)
        c += 1

    return content

def filter_data(data, filter, sample_frequency, filter_order, filter_type, verbose = 0):
    """
    Takes a list of raw signals as input, applies a band-pass filter on each of them and outputs them as a list.

    The return of this function is in the format: filtered_data.

    Parameters:
        - data: list of signals that will be band-pass filtered;
        - filter: a list with length 2, where the first value is the lowcut of the band-pass filter used in
        pre-processing, and the second value is the highcut;
        - sample_frequency: frequency of the sampling;
        - filter_order: order of the filter;
        - filter_type: type of the filter used:
            * 'sosfilt': using the sosfilt() function from the scipy library.
            * 'filtfilt': using the firwin() and filtfilt() functions from the scipy library.
    
    Optional Parameters:
        - verbose: if set to 1, prints how many % of data is currently filtered (for each interval of 10%).
        Default value is 0.
    """

    filtered_data = list()

    if verbose == 1:
        count = 0
        flag = 0
        print('Data is being filtered: 0%...',end='')

    for signal in data:
        # print('filter[0], filter[1], sample_frequency, filter_order, filter_type')
        # print(filter[0], filter[1], sample_frequency, filter_order, filter_type)
        filtered_data.append(pre_processing(signal, filter[0], filter[1], sample_frequency, filter_order, filter_type))

        if verbose == 1:
            count += 1
            flag = utils.verbose_each_10_percent(count, len(data), flag)
    
    return filtered_data

def normalize_signal(content, normalize_type):
    """
    Normalizes an EEG signal.

    Parameters:
        - content: the EEG signal that will be normalized.
        - normalize_type: type of normalization used:
            * 'each_channel': each channel of the EEG signal will be used to compute the mean, standard deviation,
            minimum and maximum values, which will be applied only to themselves in order to normalize them.
            * 'all_channels': all channels of the EEG signal will be used to compute the mean, standard deviation,
            minimum and maximum values, which will be applied to each signal in order to normalize them.
    """
    
    channels = content.shape[0]
    c = 0
    
    if(normalize_type == 'each_channel'):
        while c < channels:
            content[c] -= np.mean(content[c])
            content[c] += np.absolute(np.amin(content[c]))
            content[c] /= np.std(content[c])
            content[c] /= np.amax(content[c])

            c += 1
    elif(normalize_type == 'all_channels'):
        content -= np.mean(content)

        min_value = np.amin(content)
        while c < channels:
            content[c] += np.absolute(min_value)
            c += 1
        c = 0

        standard_deviation = np.std(content)
        while c < channels:
            content[c] /= standard_deviation
            c += 1
        c = 0

        max_value = np.amax(content)
        while c < channels:
            content[c] /= max_value
            c += 1
        c = 0
    elif(normalize_type == 'sun'):
        while c < channels:
            mean = np.mean(content[c])
            std = np.std(content[c])

            content[c] -= mean
            content[c] /= std

            c += 1
    else:
        print('ERROR: Invalid normalize_type parameter.')

    return content

def normalize_data(data, normalize_type, verbose = 0):
    """
    Takes a list of signals as input, normalizes and outputs them as a list.

    The return of this function is in the format: normalized_data.

    Parameters:
        - data: list of signals that will be normalized;
        - normalize_type: type of normalization used:
            * 'each_channel': each channel of the EEG signal will be used to compute the mean, standard deviation,
            minimum and maximum values, which will be applied only to themselves in order to normalize them.
            * 'all_channels': all channels of the EEG signal will be used to compute the mean, standard deviation,
            minimum and maximum values, which will be applied to each signal in order to normalize them.
    
    Optional Parameters:
        - verbose: if set to 1, prints how many % of data is currently filtered (for each interval of 10%).
        Default value is 0.
    """

    normalized_data = list()

    if verbose == 1:
        count = 0
        flag = 0
        print('Data is being normalized: 0%...',end='')

    for signal in data:
        normalized_data.append(normalize_signal(signal, normalize_type))

        if verbose == 1:
            count += 1
            flag = utils.verbose_each_10_percent(count, len(data), flag)

    return normalized_data