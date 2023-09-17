import numpy as np

class MultivariateDecisionTree:
    """A dict-based representation of a multivariate decision tree.
    
    The binary tree is represented as a number of dicts. The element at key t
    of each dict holds information about the node t. Node 1 is the tree's root,
    and for an arbitrary branch node t, its left child is 2t and its right
    child is 2t+1.
    
    Attributes
    ----------
    coef : dict
        coef[t] holds the feature coefficients of the branching hyperplane of
        branch node t.
    
    intercept : dict
        intercept[t] holds the intercept (a.k.a. bias) of the branching
        hyperplane of branch node t.
    
    label : dict
        label[t] holds the label of leaf node t.
    """
    def __init__(self):
        self.coef = {}
        self.intercept = {}
        self.label = {}
    
    def predict(self, X):
        """Predict target for X.
        
        Parameters
        ----------
        X : NumPy ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        y : NumPy ndarray of shape (n_samples,)
        """
        n_samples = X.shape[0]
        y_dtype = np.min_scalar_type(list(self.label.values()))
        y = np.empty(n_samples, dtype=y_dtype)
        for i in range(n_samples):
            t = 1
            while t in self.coef:
                t = 2*t + (np.dot(self.coef[t], X[i]) > self.intercept[t])
            y[i] = self.label[t]
        return y
