import numpy as np
from numpy.linalg import norm

eps = np.finfo(float).eps


class SVM:
    """ Models a Support Vector machine classifier based on the PEGASOS algorithm. """

    def __init__(self, n_epochs, lambDa, use_bias=True):
        """ Constructor method """

        # weights placeholder
        self._w = None
        self._original_labels = None
        self._n_epochs = n_epochs
        self._lambda = lambDa
        self._use_bias = use_bias

    def map_y_to_minus_one_plus_one(self, y):
        """
        Map binary class labels y to -1 and 1
        """
        ynew = np.array(y)
        self._original_labels = np.unique(ynew)
        assert len(self._original_labels) == 2
        ynew[ynew == self._original_labels[0]] = -1.0
        ynew[ynew == self._original_labels[1]] = 1.0
        return ynew

    def map_y_to_original_values(self, y):
        """
        Map binary class labels, in terms of -1 and 1, to the original label set.
        """
        ynew = np.array(y)
        ynew[ynew == -1.0] = self._original_labels[0]
        ynew[ynew == 1.0] = self._original_labels[1]
        return ynew

    def loss(self, y_true, y_pred):
        """
        The PEGASOS loss term

        Parameters
        ----------
        y_true: np.array
            real labels in {0, 1}. shape=(n_examples,)
        y_pred: np.array
            predicted labels in [0, 1]. shape=(n_examples,)

        Returns
        -------
        float
            the value of the pegasos loss.
        """

        """
        Write HERE the code for computing the Pegasos loss function.
        """
        #LOSS:quanto sta sbagliando il modello
        #questa funzione di loss deve essere più vicina allo 0 possibile se cosi i dati sono classificati bene
        #e rispettare il vincolo che i punti siano oltre il margine
        #la prima parte della formula e la parte di regolarizazzione in cui moltiplico
        #lamda il parametro di regolarizazzione per i pesi per andare a evitare che il modello
        #impari troppo bene i dati di trainig(overfitting) andando a diminure i valori dei pesi troppo grandi
        #con un lamda alto permetto al modello di rimanare facile e di non adattarsi ai dati di training richiando
        #la seconda parte e la hinge loss si occupa di vedere se i punti sono stati classificati bene
        #se y_true*(X*w) e >= 1 allora la classificazione e stata fatta bene  e rispetta il vincolo
        #la situazione perfetta e quella in cui ho una lost che dipende solo dalla prima parte
        loss=((self._lambda/2)*np.linalg.norm(self._w)**2)+np.mean(np.maximum(0,1-y_true*y_pred))
        return loss

    def fit_gd(self, X, Y, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        verbose: bool
            whether or not to print the value of cost function.
        """

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0],1),dtype=X.dtype)], axis=-1)

        n_samples, n_features = X.shape
        Y = self.map_y_to_minus_one_plus_one(Y)

        # initialize weights
        self._w = np.zeros(shape=(n_features,), dtype=X.dtype)

        t = 0
        # loop over epochs
        for e in range(1, self._n_epochs+1):
            #per ogni elemento del dataset di training vado a calcolare la hinge loss e man mano aggiorno i pesi
            #andando piano a piano ad arrivare ai pesi ottimali che mi massimizzano la distanza tra i punti e il margine
            #e mi classifichino bene i punti
            for j in range(n_samples):
                """
                Write HERE the update step.
                """
                t=t+1
                mu=1/(t*self._lambda)
                #se la hinge loss  e < 1 significa che il punto non e classificato bene e devo modificare i pesi
                if np.dot(Y[j],np.dot( X[j],self._w))<1:
                    #questa e la formula di aggiornamento dei pesi di PEGASOS
                    #il livello di correzione dipende da quanto la hinge loss e lontana da 1
                    self._w=(1-mu*self._lambda)*self._w+mu*(Y[j]*X[j])
                else:
                    #se la hinge loss >=1 significa che il punto e stato classificato bene e rispetta i vincoli del margine
                    #in questo caso molto semplicemente non modifico i pesi ma li vado solo a regolarizzare i pesi per
                    #evitare problemi di overfitting
                    self._w=(1-mu*self._lambda)*self._w
                pass
            
            # predict training data
            cur_prediction = np.dot(X, self._w)

            # compute (and print) cost
            cur_loss = self.loss(y_true=Y, y_pred=cur_prediction)

            if verbose:
                print("Epoch {} Loss {}".format(e, cur_loss))

    def predict(self, X):

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0],1),dtype=X.dtype)], axis=-1)

        """
        Write HERE the criterium used during inference. 
        W * X > 0 -> positive class
        X * X < 0 -> negative class
        """
        #ottengo un vettore dal prodotto scalare tra il vettore dei  pesi w e i dati del dataset
        #ogni val in vettore e dato da prod scalare vettore pesi w e vettore X[i] dato
        #e usando where che una specie di if se questo valore e >0 appartiene alla classe +
        #se non appartiene alla classe -
        return np.where(np.dot(self._w,X.T) > 0.0,
                        self._original_labels[1], self._original_labels[0])

