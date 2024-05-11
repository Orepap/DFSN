import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


def DFSN(df, threshold, epochs, batch_size, verbose, plot):


    X_train = df.iloc[:, :-1]
    Y_train = df.iloc[:, -1]


    model = Sequential([
        Dense(64, activation='relu', input_shape=(np.array(X_train).shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    scaler_cnn = MinMaxScaler()
    X_train_scaled_cnn = scaler_cnn.fit_transform(X_train)

    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled_cnn, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Access the weights of the connections between input and last hidden layer
    input_to_hidden_weights = model.layers[0].get_weights()[0]

    gene_stds = []
    for gene_importance_vector in np.array(input_to_hidden_weights):
        gene_stds.append(np.std(gene_importance_vector))

    ranked_indices_stds = np.argsort(gene_stds)


    top_gs_ind_stds = ranked_indices_stds[-threshold:]

    random_integers = np.random.randint(0, len(input_to_hidden_weights), size=10)
    ii = top_gs_ind_stds[:threshold]


    if plot:

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.05])

        vmin = min(np.min(input_to_hidden_weights[ii]), np.min(input_to_hidden_weights[random_integers]))
        vmax = max(np.max(input_to_hidden_weights[ii]), np.max(input_to_hidden_weights[random_integers]))

        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(input_to_hidden_weights[ii], cmap='viridis', annot=False, cbar=False, vmin=vmin, vmax=vmax, ax=ax1)
        ax1.set_title('Extracted features', fontsize=16)
        ax1.set_xticks([])
        ax1.set_yticklabels([list(df.columns)[f] for f in ii])

        ax2 = fig.add_subplot(gs[1, 0])
        sns.heatmap(input_to_hidden_weights[random_integers], cmap='viridis', annot=False, cbar=False, vmin=vmin, vmax=vmax, ax=ax2)
        ax2.set_title('Random features', fontsize=16)
        ax2.set_xticks([])
        ax2.set_yticklabels([list(df.columns)[f] for f in random_integers])

        cax = fig.add_subplot(gs[:, 1])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cax, orientation='vertical')

        plt.tight_layout()

        plt.show()

    return [list(df.columns)[f] for f in ii]
