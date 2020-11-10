# SubGattPool

Subgatt:


This is a tensorflow based implementation of Subgraph Attention as discussed in the paper.

Dataset:
1.  The dataset_class folder contains all the datasets which we used in experiments of graph classification.


How to run: 
1) For Graph Classification: (Default dataset is set to MUTAG)
	python graphclassification.py


Requirements:
1) python (version 3.6 or above)
2) tensorflow (version 1.13)
3) networkx
4) keras
5) numpy
6) pickle
7) scipy
8) pandas
9) collections



Parameters: 
1) For Graph Classification:
	dataset: The name of the dataset
	epoch: Number of epochs to train the mdoel
	sub_samp: Number of subgraph samples for each node
	sub_leng: The maximum length of any subgraph
	pool_rt: Pooling ratio
	pool_lay: Number of SubGattPool layers
	sub_lay: Number of SubGatt attention layer
	learning_rate: Learning rate
	embd_dim: Embedding dimension


We can specify these parameters while running these python files.
	For eg: To specify any other dataset, run following command: 
	python graph_classification.py --dataset NCI1

	
