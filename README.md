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

	
