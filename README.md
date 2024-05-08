# DFSN: Deep Feature Selection Network


DFSN is a deep neural network for feature selection in a binary classification problem. 


Authors:  

• *Orestis D. Papagiannopoulos*  

Unit of Medical Technology and Intelligent Information Systems  
Dept. of Materials Science and Engineering
University of Ioannina   
Ioannina, Greece

Contact: *orepap@uoi.gr*


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

df = "..." # path to the data file
threshold = "..." # path to the correlation file
epochs = # Specify the number of neurons
batch_size
verbose
plot

selected_features = DFSN(df=df, threshold=threshold, epochs=epochs, batch_size=batch_size, verbose=verbose, plot=plot)
```

The code returns a list with the selected features.
