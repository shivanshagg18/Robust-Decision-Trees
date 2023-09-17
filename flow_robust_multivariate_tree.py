import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import gurobipy as gp
from gurobipy import GRB
from .decision_tree import MultivariateDecisionTree

class RBT(ClassifierMixin, BaseEstimator):
    """Our implementation of FlowOCT (Aghaei et al. (2021)).
    
    This implementation follows the formulation (7). We do not include
    the penalty term in the objective (no lambda), instead we use the
    sparsity constraint to restrict the number of branching nodes.
    
    Parameters
    ----------
    time_limit : positive float, default=None
        Training time limit.
    
    verbose : bool, default=False
        Enables or disables Gurobi console logging for the MIP.
    
    Attributes
    ----------
    model_ : Gurobi Model
        The MIP model.
    
    decision_tree_ : UnivariateDecisionTree
        The trained decision tree.
    
    fit_time_ : float
        Time (in seconds) taken to fit the model.
    """

    def __init__(self, time_limit=None, verbose=False):
        self.time_limit = time_limit
        self.verbose = verbose

    def fit(self, X, y):
        """Train a decision tree using a MIP model.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of shape (n_samples, n_features)
            The training input samples.
        
        y : pandas Series or NumPy ndarray of shape (n_samples,)
            The target values (class labels).
        
        Returns
        -------
        self : RBT
            Fitted estimator.
        """

        start_time = time.perf_counter()

        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        # Create MIP model, set Gurobi parameters, warm start
        self.model_ = self._mip_model(X, y)
        self._set_gurobi_params()
        
        # Solve MIP model
        callback = RBT._callback
        self.model_.optimize(callback)
        
        # Construct the decision tree
        self.decision_tree_ = self._construct_decision_tree()
        
        self.fit_time_ = time.perf_counter() - start_time
        
        return self
    
    def predict(self, X):
        """Predict class labels for samples.
        
        Parameters
        ----------
        X : pandas DataFrame or NumPy ndarray of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y : pandas Series of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self)
        index = None
        if isinstance(X, pd.DataFrame):
            index = X.index
        X = check_array(X) # Converts to ndarray
        y = self.decision_tree_.predict(X)
        y = pd.Series(y, index=index)
        return y
    
    def _mip_model(self, X, y):
        """Create the MIP model.
        
        Parameters
        ----------
        X : NumPy ndarray of shape (n_samples, n_features)
            The training input samples.
        
        y : NumPy ndarray of shape (n_samples,)
            The target values (class labels).
        
        Returns
        -------
        model : Gurobi Model
            The MIP model. Has the following data: `_X_y`,
            `_flow_graph`, `_vars`.
        """

        n_samples, n_features = X.shape
        classes = unique_labels(y)
        
        branch_nodes = list(range(1, 2**self.max_depth))
        leaf_nodes = list(range(2**self.max_depth, 2**(self.max_depth+1)))
        arcs = (
            [('s', 1)]
            + [(n, 2*n) for n in branch_nodes]
            + [(n, 2*n+1) for n in branch_nodes]
            + [(n, 't') for n in branch_nodes+leaf_nodes]
        )
        ancestors = {} # dict mapping each node to a list of its ancestors
        for n in branch_nodes+leaf_nodes:
            a = []
            curr_node = n
            while curr_node > 1:
                parent = int(curr_node/2)
                a.append(parent)
                curr_node = parent
            ancestors[n] = a
        
        model = gp.Model()
        model._X_y = X, y
        model._flow_graph = branch_nodes, leaf_nodes, arcs

        c = model.addVars(leaf_nodes, classes, vtype=GRB.BINARY)
        a = model.addVars(branch_nodes, n_features)
        a_cap = model.addVars(branch_nodes, n_features)
        b = model.addVars(branch_nodes, lb=-1, ub=1)

        t = model.addVars(n_samples, ub=1)

        obj_fn = t.sum()
        model.setObjective(obj_fn, GRB.MAXIMIZE)

        model.addConstrs((
            c.sum(n, "*") == 1
            for n in leaf_nodes
        ))

        model.addConstrs((
           t.sum() <= 1
        ))

        model.addConstrs((
            a_cap.sum(n, "*") <= 1
            for n in branch_nodes
        ))

        model.addConstrs((
            a_cap[n, i] >= a[n, i]
            for n in branch_nodes
            for i in  n_features
        ))

        model.addConstrs((
            a_cap[n, i] >= -a[n, i]
            for n in branch_nodes
            for i in  n_features
        ))