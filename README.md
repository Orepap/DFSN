# DFSN: Deep Feature Selection Network


DFSN is a deep neural network for feature selection in a binary classification problem. 


Authors:  

• *Orestis D. Papagiannopoulos*  

Unit of Medical Technology and Intelligent Information Systems  
Dept. of Materials Science and Engineering
University of Ioannina   
Ioannina, Greece

Contact: *orepap@uoi.gr*


**PAPER**:   
Papagiannopoulos, Orestis D., Costas Papaloukas, and Dimitrios I. Fotiadis. "Deep Learning for Biomarkers Discovery in Auto-inflammatory Disorders." 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology. IEEE, 2023.

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
