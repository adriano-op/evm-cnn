# Description
With technological advancements enhancing the capabilities of mobile devices to capture physiological data, new human characteristics are increasingly being explored for the development of biometric systems. The remote cardiac signal, obtained through the Eulerian Video Magnification (EVM) technique from image sequences, emerges as a promising physiological trait in this evolving landscape. This study aims to assess the viability of the remote cardiac signal as a biometric modality for verification tasks. The methodology includes data collection in a controlled environment, transforming video sequences into one-dimensional signals using the EVM technique, filtering unwanted frequencies from the signal, and employing a simple convolutional neural network for data representation. A one-against-all protocol for biometric verification evaluation complements this approach. Experiments on a dataset containing videos of 56 individuals yielded an Equal Error Rate (EER) of 13%.


## Methodology Applied to Data Collection
![modelagem_coleta](https://github.com/user-attachments/assets/c7d97dc8-c0e2-4036-92b4-18f7469d0a42)

 


## Methodology Applied to EVM
![evm_ingles](https://github.com/user-attachments/assets/6f114036-df7c-4720-b44b-1a0fadd5af6f)

## Methodology Applied to CNN
![modelagem_cnn_novo_1](https://github.com/user-attachments/assets/fb58d074-a139-4d06-bcf9-7c2b2a11d5aa)

# Getting Started
## Dependencies
The quickest way to get started is to install the [TensorFlow](https://www.tensorflow.org/) distribution.

### Complementary Libraries and Frameworks
- Keras
- TensorBoard
- OpenCV
- NumPy
- Pandas
- scikit-learn (sklearn)
- Matplotlib

# Basic Usage
The `DataSet_CSV_V2` folder contains data from 56 volunteers, recorded in 2 sessions. Each volunteer has a folder with 2 files: file 1 corresponds to session 1 and file 2 to session 2.

To run the application, follow the steps below:

1. **Open the `EVM_biometrics.py` file and correct the `folder_path` and `processed_data_path` variables by providing the path to the `Dataset_CSV_V2` folder.**

2. **Adjust the hyperparameters as needed, such as `training_epochs` and `initial_learning_rate`.**

3. **The variables `train_tasks = [1]` and `test_tasks = [2]` are responsible for training (`train_tasks`) and testing (`test_tasks`). With this configuration, the training data will not be used in the test.**

When you run the algorithm, the program will load all the files, apply the filter in the range of [0.01, 29] Hz, and then perform the `train_test_split`, separating 10% of the data for testing. Subsequently, the `MinMaxScaler` will be applied to the `x_train`, `x_val`, `y_train`, `y_val`, `x_test`, and `y_test` variables.

After execution, the model will be saved in Keras format, and the processing results will be generated, including: EER, genuine and impostor curves of the raw and processed data, EER graph, average ROC curve graph, and ROC curve graph for each class.

# Command-Line Options

```python
parser.add_argument('--datagen', action='store_true', help='The model will use data generators to crop data on the fly')
parser.add_argument('--nofit', action='store_true', help="model.fit will not be executed. The weights will be loaded
from the 'model_weights.keras' file, which is generated if you have run the model in Identification mode at least once")

parser.add_argument('--noimode', action='store_true', help="The model won't run in Identification Mode")
parser.add_argument('--novmode', action='store_true', help="The model won't run in Verification Mode")
parser.add_argument('--training_epochs', type=int, required=True, help='Total number of training epochs')



Without data generators:

In the terminal, run the file: python3 /media/work/adrianoandrade/redeNeural/EVM_biometrics.py --training_epochs 100.
This will execute with --datagen=False (without data generator) and will also run in noimode and novmode.

With data generator:

In the terminal, run the file: python3 /media/work/adrianoandrade/redeNeural/EVM_biometrics.py --datagen --training_epochs 100.

Without verification mode:

In the terminal, run the file: python3 /media/work/adrianoandrade/redeNeural/EVM_biometrics.py --novmode --training_epochs 100.
 
```
# Documentation
The full article of the   is available at  []


	
