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
verbose = "..." # Training verbosity (0, 1 or 2)
plot = "..." # Boolean to show or not the weights heatmap (True or False)

selected_features = DFSN(df=df, threshold=threshold, epochs=epochs, batch_size=batch_size, verbose=verbose, plot=plot)
```

**Caution!**    
The dataframe (df) should contain the samples as rows, the features as columns AND a last column specifying the binary labels (0 and 1).


The code returns a list with the selected features.
   


# HOW IT WORKS
<p align="justify">
Features demonstrating substantial weight deviations across the input (feature) layer and the first hidden dense layer, could hold higher significance in training the neural network and potentially possess heightened inherent relevance. Utilizing these connection weights, the features with the highest shifts in weight contributions (measured by standard deviation) are selected. This is due to the fact that they expedite the convergence of the neural network training, in contrast to random genes in the dataset.  </p>
