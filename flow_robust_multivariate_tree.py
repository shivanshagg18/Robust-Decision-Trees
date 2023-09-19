import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import gurobipy as gp
from gurobipy import GRB
from .decision_tree import MultivariateDecisionTree

class RDT(ClassifierMixin, BaseEstimator):
    """
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
        self : RDT
            Fitted estimator.
        """

        start_time = time.perf_counter()

        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        # Create MIP model, set Gurobi parameters, warm start
        self.model_ = self._mip_model(X, y)
        self._set_gurobi_params()
        
        # Solve MIP model
        callback = RDT._callback
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

        model._vars = c, a, a_cap, b, t

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

    def _set_gurobi_params(self):
        """Set Gurobi parameters."""
        model = self.model_
        model.Params.LogToConsole = self.verbose
        model.Params.LazyConstraints = True
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = 0.0
        # Need to reduce desired gap (exactly 1) by a small amount due to numerical issues
        model.Params.MIPGapAbs = 0.999

    @staticmethod
    def _callback(model, where):
        """Gurobi callback. Adds lazy cuts."""
        if where == GRB.Callback.MIPSOL:
            c, a, a_cap, b, t = model._vars
            a_vals = model.cbGetSolution(a)
            b_vals = model.cbGetSolution(b)
            c_vals = model.cbGetSolution(c)
            X, y = model._X_y
            branch_nodes, leaf_nodes, arcs = model._flow_graph
            n_samples, n_features = X.shape
            classes = unique_labels(y)

            lam = np.array([0.1 for i in n_features])
            budget = 10

            mu = np.array([[i, 0] for i in n_samples])
            perturb = np.array([[0 for j in n_features] for i in n_samples])

            for i, x in enumerate(X):
                j = 1
                while j <= len(branch_nodes):
                    if np.dot(a_vals[j], x) > b_vals[j]:
                        j = 2*j + 1
                    else:
                        j = 2*j
                if c_vals[j, y[i]] != 1:
                    continue

                min_mu = 1000000
                for k in leaf_nodes:
                    if np.argmax(c_vals[k]) == y[i]:
                        continue

                    sub_model = gp.Model()
                    perturb_var = sub_model.addVars(n_features)
                    perturb_cap_var = sub_model.addVars(n_features)

                    obj_fn = np.dot(lam, perturb_cap_var)
                    sub_model.setObjective(obj_fn, GRB.MINIMIZE)

                    sub_model.addConstrs((
                        perturb_cap_var[n] >= perturb_var[n]
                        for n in n_features
                    ))

                    sub_model.addConstrs((
                        perturb_cap_var[n] >= perturb_var[n]
                        for n in n_features
                    ))

                    parent = k // 2
                    child = k % 2
                    while parent >= 1:
                        if child == 0:
                            sub_model.addConstrs((
                                np.dot(a_vals[parent], perturb_var) < b_vals[parent] - np.dot(a_vals[parent], x)
                            ))
                        else:
                            sub_model.addConstrs((
                                np.dot(a_vals[parent], perturb_var) >= b_vals[parent] - np.dot(a_vals[parent], x)
                            ))
                        
                        parent = parent // 2
                        child = parent % 2

                    sub_model.optimize()

                    opt_sol = sub_model.getVars()
                    if sub_model.objVal <= min_mu:
                        min_mu = sub_model.objVal
                        mu[i, 1] = min_mu
                        for v in opt_sol:
                            if v.varName == "perturb_var":
                                perturb[i] = v.x
            
            mu = mu[mu[:, 1].argsort()]
            perturb_set = np.array([])
            total_cost = 0
            for i in n_samples:
                if total_cost + mu[i, 1] <= budget:
                    total_cost += mu[i, 1]
                    perturb_set.append(mu[i, 0])
                else:
                    break
            
            