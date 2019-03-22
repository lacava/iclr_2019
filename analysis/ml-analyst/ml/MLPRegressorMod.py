from sklearn.neural_network import MLPRegressor
import numpy as np

# define a fn to output last hidden layer of network    
class MLPRegressorMod(MLPRegressor):
    def transform(self, X):
        """Output the last layer of the trained model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
        Returns
        -------
        phi : array-like, shape (n_samples,n_layers in final layer) 
        """

        # X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + \
        [self.n_outputs_]

        # Initialize layers
        activations = [X]

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations)
        phi = activations[-2]
        return phi 
