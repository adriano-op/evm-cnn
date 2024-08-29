# Description
With technological advancements enhancing the capabilities of mobile devices to capture physiological data, new human characteristics are increasingly being explored for the development of biometric systems. The remote cardiac signal, obtained through the Eulerian Video Magnification (EVM) technique from image sequences, emerges as a promising physiological trait in this evolving landscape. This study aims to assess the viability of the remote cardiac signal as a biometric modality for verification tasks. The methodology includes data collection in a controlled environment, transforming video sequences into one-dimensional signals using the EVM technique, filtering unwanted frequencies from the signal, and employing a simple convolutional neural network for data representation. A one-against-all protocol for biometric verification evaluation complements this approach. Experiments on a dataset containing videos of 56 individuals yielded an Equal Error Rate (EER) of 13\%

## methodology applied to data collection

![modelagem_cnn_novo_1](https://github.com/user-attachments/assets/fb58d074-a139-4d06-bcf9-7c2b2a11d5aa)


# redeNeural
A pasta DataSet_CSV_V2, possui dados dos 56 voluntários, gravados em 2 sessões.

Para executar a aplicação, siga os seguintes passos:

	
	* Abra o arquivo CNN_EVM.ipynb (arquivo main) e corrija as variaveis "folder_path" e "processed_data_path", 
 		passando a url  e onde se encontra a pasta "Dataset_CSV_V_2";

	* Ajuste os hiperparamteros que achar necessário, como training_epochs, initial_learning_rate.

 	* As variáveis: train_tasks = [1] e test_tasks = [2], serão responsáveis pelo treinamento (train_tasks)  e teste (test_tasks).
  	  Com essa definição, os dados de treino, não serão visttos no teste.

	Em seguida, mude a "RunTime Type", para T4 GPU, caso esteja usando o google colab.
 	O algoritmo está pronto para execução.

Após a execução, o modelo será salvo na extensão H5 e será gerado os resultados processamento, o que inclui
a acurária e os graficos de aprendizado.

	
