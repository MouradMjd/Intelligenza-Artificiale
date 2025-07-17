"""
Eigenfaces main script.
"""

import numpy as np

from utils import show_eigenfaces
from utils import show_nearest_neighbor
from data_io import get_faces_dataset

import matplotlib.pyplot as plt
plt.ion()


class Eigenfaces:
    """
    Performs PCA to project faces in a reduced space.
    """

    def __init__(self, n_components: int):
        """
        Parameters
        ----------
        n_components: int
            number of principal component
        """
        self.n_components = n_components

        # Per-feature empirical mean, estimated from the training set.
        self._x_mean = None

        # Principal axes in feature space, representing the directions
        # of maximum variance in the data.
        self._ef = None

    def fit(self, X: np.ndarray, verbose: bool = False):
        """
        Parameters
        ----------
        X: ndarray
            Training set will be used for fitting the PCA
                (shape: (n_samples, w*h))
        verbose: bool
        """
        # compute mean vector and store it for the inference stage su colonne
        self._x_mean =np.mean(X,axis=0)

        if verbose:
            # show mean face
            plt.imshow(np.reshape(self._x_mean, newshape=(112, 92)), cmap='gray')
            plt.title('mean face')
            plt.waitforbuttonpress()
            plt.close()

        # remove mean from the data
        X_norm = X-self._x_mean

        # compute covariance with the eigen-trick
        #perche uso eigen-trick: lo uso visto che ottenere gli autovalori e autovettori
        #della grande matrice di covarianza N_feature x N_feature(X_norm.T @ X_norm) e molto complicato
        #se il numero di feature e grande quindi passo a lavorare con una matrice(X_norm @ X_norm.T)
        #N_samples x N_samples molto piu fattibile visto che
        #gli autovettori e autovalori  di questa sono gli stessi della matrice originale
        cov = X_norm@X_norm.T

        # compute eigenvectors of the covariance matrix
        eigval, eigvec = np.linalg.eig(cov)

        # sort them (decreasing order w.r.t. eigenvalues)
        #sorted_eigvec ha shape (n_samples,n_samples)
        #usando [::-1] inverto e ho dal più grande al più piccolo
        inx=np.argsort(eigval)[::-1]
        sorted_eigval, sorted_eigvec = eigval[inx], eigvec[:,inx]

        # select principal components (vectors)
        #principal_sorted_eigvec ha shape(n_samples, n_components) ogni cella di una colonna si riferisce a un immagine
        #uso [:,:self.n_components] per prendere le prime k colonne visto che gli auto vettoni sono disposti in colonne
        principal_sorted_eigval, principal_sorted_eigvec = sorted_eigval[:self.n_components], sorted_eigvec[:,:self.n_components]
        #prendo le colonne da 0 a n_components


        # retrieve original eigenvec
        #riottengo gli autovettori della matrice originale da quelli che ho calcolato per
        #ottenere una matrice (n_samples,n_features) visto che principal_sorted_eigvec (n_samples, n_components)
        #per ottenere una colonna quindi un autovettore della matrice originale basta
        #moltiplicare la j-esima colonna della principal_sorted_eigvec per ogni riga della matrice X_norm.T
        #alla fine avro una matrice self._ef con shape (n_features,n_componenti) in cui ogni colonna e una autovettore
        self._ef = np.dot(X_norm.T, principal_sorted_eigvec)

        # normalize the retrieved eigenvectors to have unit length
        self._ef = self._ef/principal_sorted_eigval**1/2

        if verbose:
            # show eigenfaces
            show_eigenfaces(self._ef, (112, 92))
            plt.waitforbuttonpress()
            plt.close()

    def transform(self, X: np.ndarray):
        """
        Parameters
        ----------
        X: ndarray
            faces to project (shape: (n_samples, w*h))
        Returns
        -------
        out: ndarray
            projections (shape: (n_samples, self.n_components)
        """

        # project faces according to the computed directions
        return (X - self._x_mean) @ self._ef

    def inverse_transform(self, X: np.ndarray):
        return (X + self._x_mean) @ self._ef.T


class NearestNeighbor:

    def __init__(self):
        self._X_db, self._Y_db = None, None

    def fit(self, X: np.ndarray, y: np.ndarray,):
        """
        Fit the model using X as training data and y as target values
        """
        self._X_db = X
        self._Y_db = y

    def predict(self, X: np.ndarray):
        """
        Finds the 1-neighbor of a point. Returns predictions as well as indices of
        the neighbors of each point.
        """
        num_test_samples = X.shape[0]

        # predict test faces
        predictions = np.zeros((num_test_samples,))
        nearest_neighbors = np.zeros((num_test_samples,), dtype=np.int32)

        for i in range(num_test_samples):

            distances = np.sum(np.square(self._X_db - X[i]), axis=1)

            # nearest neighbor classification
            nearest_neighbor = np.argmin(distances)
            nearest_neighbors[i] = nearest_neighbor
            predictions[i] = self._Y_db[nearest_neighbor]

        return predictions, nearest_neighbors


def main():

    # get_data
    X_train, Y_train, X_test, Y_test = get_faces_dataset(path='att_faces', train_split=0.6)

    # number of principal components
    n_components = 15

    # fit the PCA transform
    eigpca = Eigenfaces(n_components)
    eigpca.fit(X_train, verbose=True)

    # project the training data
    proj_train = eigpca.transform(X_train)

    # project the test data
    proj_test = eigpca.transform(X_test)

    # fit a 1-NN classifier on PCA features
    nn = NearestNeighbor()
    nn.fit(proj_train, Y_train)

    # Compute predictions and indices of 1-NN samples for the test set
    predictions, nearest_neighbors = nn.predict(proj_test)

    # Compute the accuracy on the test set
    test_set_accuracy = float(np.sum(predictions == Y_test)) / len(predictions)
    print(f'Test set accuracy: {test_set_accuracy}')

    # Show results.
    show_nearest_neighbor(X_train, Y_train,
                          X_test, Y_test, nearest_neighbors)


if __name__ == '__main__':
    main()
