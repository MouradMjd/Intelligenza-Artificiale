import matplotlib.pyplot as plt
import numpy as np

from utils import cmap

class WeakClassifier:
    """
    Function that models a WeakClassifier based on a simple threshold.
    """

    def __init__(self):

        # initialize a few stuff
        self._dim = None
        self._threshold = None
        self._label_above_split = None

    def fit(self, X: np.ndarray, Y: np.ndarray):

        n, d = X.shape
        possible_labels = np.unique(Y)

        """ pick a random feature (see np.random.choice) """
        self._dim = np.random.choice(d) #TODO!

        """ pick a random threshold (see np.random.uniform)
            NB: look at the interval [min,max] from the selected dimension """
        self._threshold =  np.random.uniform(np.min(X[:,self._dim]),
                                                    np.max(X[:,self._dim]))#TODO!

        """ pick a random verse (see np.random.choice)
            case a) feature >= _threshold ==>> then predict 1
            case b) feature >= _threshold ==>> then predict -1 """
        self._label_above_split = np.random.choice(possible_labels) #TODO!

    def predict(self, X: np.ndarray):

        num_samples = X.shape[0]

        """ fill y_pred with the predictions """
        y_pred = np.zeros(shape=num_samples)
        for i in range(num_samples):
            #per ogni sampele controllo se il valore della feature scelta è maggiore o uguale alla soglia
            if X[i,self._dim]>= self._threshold:
                y_pred[i] = self._label_above_split
            else:
                y_pred[i] = -self._label_above_split
        return y_pred #TODO!


class AdaBoostClassifier:
    """
    Function that models a Adaboost classifier
    """

    def __init__(self, n_learners: int, n_max_trials: int = 200):
        """
        Model constructor

        Parameters
        ----------
        n_learners: int
            number of weak classifiers.
        """

        # initialize a few stuff
        self.n_learners = n_learners
        self.learners = []
        self.alphas = np.zeros(shape=n_learners)
        self.n_max_trials = n_max_trials

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
        """
        Trains the model.

        Parameters
        ----------
        X: ndarray
            features having shape (n_samples, dim).
        Y: ndarray
            class labels having shape (n_samples,).
        verbose: bool
            whether or not to visualize the learning process.
            Default is False
        """

        n, d = X.shape
        possible_labels = np.unique(Y)

        if d != 2:
            verbose = False  # only plot learning if 2 dimensional

        assert possible_labels.size == 2, 'Error: data is not binary'

        """ initialize the sample weights as equally probable """
        sample_weights = np.zeros(n) #TODO!
        np.ndarray.fill(sample_weights,1/n)


        for l in range(self.n_learners):

            """ choose the indexes of 'difficult' samples. See np.random.choice
                https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.choice.html
                Pay attention to p, which indicates the probabilities that will be used during sampling."""
            print(np.sum(sample_weights))
            cur_idx = np.random.choice(np.arange(n),size=n,replace=True,p=sample_weights) #TODO!

            # extract 'difficult' samples
            cur_X = X[cur_idx]
            cur_Y = Y[cur_idx]

              # search for a weak classifier
            error = 1
            n_trials = 0
            cur_wclass = None
            y_pred = None

            # search for a 'good' weak classifier
            while error > 0.5:

                cur_wclass = WeakClassifier()

                """ train the weak classifier on the dataset subsample """
                cur_wclass.fit(cur_X,cur_Y)#TODO!
                pass

                """ compute the predictions on the dataset subsample """
                y_pred = cur_wclass.predict(cur_X) #TODO!

                """ according to the predicitons and labels, compute the error
                    made by the current classifier (namely, cur_wclass) """
                err_mask=y_pred!=cur_Y
                error =np.sum(sample_weights[cur_idx]*err_mask)/np.sum(sample_weights[cur_idx])   #TODO!

                n_trials += 1
                if n_trials > self.n_max_trials:
                    # initialize the sample weights again
                    sample_weights = np.ones(shape=n) / n


            """ compute the efficiency of the weak classifier """
            alpha = np.log((1 - error) / error) / 2


            self.alphas[l] = alpha

            # append the learned weak classifier to the chain
            self.learners.append(cur_wclass)



            """ based on the right and wrong predictions, update sample_weights"""

            """
            cur_idx_uni=[]
            err_mask_uni=[]
            for i in range(n):
                if cur_idx[i] in cur_idx_uni:
                    continue
                else:
                    cur_idx_uni.append(cur_idx[i])
                    err_mask_uni.append(err_mask[i])
            """

            #con questo fai la stessa cosa di sopra ma in maniera più compatta
            sample_weights[cur_idx] *= np.exp(alpha * y_pred==cur_Y)
            sample_weights[cur_idx] *= np.exp(-alpha* y_pred!=cur_Y)



            sample_weights=sample_weights/np.sum(sample_weights)

            if verbose:
                self._plot(cur_X, y_pred, sample_weights[cur_idx],
                           self.learners[-1], l)


    def predict(self, X: np.ndarray):
        """
        Function to perform predictions over a set of samples.

        Parameters
        ----------
        X: ndarray
            examples to predict. shape: (n_examples, d).

        Returns
        -------
        ndarray
            labels for each examples. shape: (n_examples,).

        """
        num_samples = X.shape[0]

        """ fill y_pred with the predictions """
        pre = np.zeros(shape=num_samples)
        # ottengo da questo for un vettore con dentro per ogni elemento la somma della previsione di tutti i modelli tenedo conto
        # della loro efficenza  con *self.alphas[i] e se la somma tende a positivo e classe 1 se no -1
        for i in range(self.n_learners):
            pre+=self.learners[i].predict(X)*self.alphas[i]
        y_pred = np.sign(pre)

        return y_pred


    def _plot(self, X: np.ndarray, y_pred: np.ndarray, weights: np.ndarray,
              learner: WeakClassifier, iteration: int):

        # plot
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=weights * 50000,
                    cmap=cmap, edgecolors='k')

        M1, m1 = np.max(X[:, 1]), np.min(X[:, 1])
        M0, m0 = np.max(X[:, 0]), np.min(X[:, 0])

        cur_split = learner._threshold
        if learner._dim == 0:
            plt.plot([cur_split, cur_split], [m1, M1], 'k-', lw=5)
        else:
            plt.plot([m0, M0], [cur_split, cur_split], 'k-', lw=5)
        plt.xlim([m0, M0])
        plt.ylim([m1, M1])
        plt.xticks([])
        plt.yticks([])
        plt.title('Iteration: {:04d}'.format(iteration))
        plt.waitforbuttonpress(timeout=0.1)
