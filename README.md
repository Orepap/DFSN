# DFSN: Deep Feature Selection Network


DFSN is a deep neural network for feature selection in a binary classification problem. 


**PAPER**:   
[Papagiannopoulos, Orestis D., Costas Papaloukas, and Dimitrios I. Fotiadis. "Deep Learning for Biomarkers Discovery in Auto-inflammatory Disorders." 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology. IEEE, 2023.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10404699).

# PREREQUISITES
```python
numpy
tensorflow.keras
matplotlib
seaborn
```

# USAGE
```python
from DFSN import DFSN

df = "..." # Pandas dataframe
threshold = "..." # Number of selected features
epochs = "..." # Number of epochs for the neural network training
batch_size = "..." # Batch size for the neural network training
verbose = "..." # Training verbosity 
plot = "..." # Boolean to show or not the weights heatmap 

selected_features = DFSN(df=df, threshold=threshold, epochs=epochs, batch_size=batch_size, verbose=verbose, plot=plot)
```

**Caution!**    
The dataframe (df) should contain the samples as rows, the features as columns AND a last column specifying the binary labels (0 and 1)


The code returns a list with the selected features.
   


# HOW IT WORKS
<p align="justify">
The Euclidean distance is first computed between an input sample and all the neurons. Then, the neuron that has the smallest distance to the sample is declared as the best matching unit (BMU) and its weights along with its nearest neighbor neurons (self-organizing) are re-adjusted to closer mimic the input sample. The novelty is the introduction of matrix norms as distance concepts. Conventional distance metrics like the Euclidean, are typically calculated between vectors. In Clust3D, where the data points are matrices, the distance between two data points is defined as the mathematical norm of the matrix of their differences. As such, Clust3D introduces the capability to train the neural network given the input samples and the neurons as matrices and not just as vectors, containing both the temporal and the spatial information. Thus, the clustering can be implemented directly on the patients, given the different timepoints altogether. </p>
