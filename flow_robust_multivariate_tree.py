import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import gurobipy as gp
from gurobipy import GRB
from decision_tree import MultivariateDecisionTree
import random

# TODO (not urgent) even though index variables can be named anything, I strongly recommend sticking to the following conventions:
# - Decision tree nodes: t (note that the decision variables are also named t)
# - Datapoints: i
# - Features: j
# - Class labels: k
# flow_oct.py uses n instead of t to index nodes because the original paper uses n to index nodes

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

    def __init__(self, time_limit=None, verbose=0, epsilon=0.01, max_depth=1):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.verbose = verbose
        self.epsilon = epsilon

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
        self.model_.optimize()
        self.cuts()
        
        # Construct the decision tree
        # self.decision_tree_ = self._construct_decision_tree()
        
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
        nodes = list(range(1, 2**(self.max_depth+1)))
        arcs = (
            [('s', 1)]
            + [(n, 2*n) for n in branch_nodes]
            + [(n, 2*n+1) for n in branch_nodes]
            + [(n, 't') for n in branch_nodes+leaf_nodes]
        )
        # dicts mapping each leaf node to its left/right-branch ancestors
        left_ancestors = {}
        right_ancestors = {}
        for t in leaf_nodes:
            la = []
            ra = []
            curr_node = t
            while curr_node > 1:
                parent = int(curr_node/2)
                if curr_node == 2*parent:
                    la.append(parent)
                else:
                    ra.append(parent)
                curr_node = parent
            left_ancestors[t] = la
            right_ancestors[t] = ra
        
        model = gp.Model()
        model._X_y = X, y
        model._flow_graph = branch_nodes, leaf_nodes, nodes, arcs
        model._ancestors = left_ancestors, right_ancestors

        c = model.addVars(leaf_nodes, classes, vtype=GRB.BINARY, name="c")
        a = model.addVars(branch_nodes, n_features, lb=-1, ub=1, name="a")
        a_cap = model.addVars(branch_nodes, n_features, lb=0, ub=1, name="a_cap")
        b = model.addVars(branch_nodes, lb=-1, ub=1, name="b")

        g = model.addVars(n_samples, vtype=GRB.BINARY, ub=1, name="t")

        model._vars = c, a, a_cap, b, g

        for label in classes:
            if t % len(classes) == label:
                c[t, label].Start = 1
            else:
                c[t, label].Start = 0

        for n in branch_nodes:
            for i in range(n_features):
                a_cap[n, i].Start = 1/n_features
                a[n, i].Start = 1/n_features

        # for n in branch_nodes:
        #     b[n].Start = random.uniform(0, 1)

        lam = np.array([1 for i in range(n_features)])
        budget = 0.5
        model._cost = lam, budget
        model._epsilon = self.epsilon

        obj_fn = g.sum()
        model.setObjective(obj_fn, GRB.MAXIMIZE)

        model.addConstrs((
            c.sum(n, "*") == 1
            for n in leaf_nodes
        ))

        # model.addConstrs((
        #     c.sum("*", n) >= 1
        #     for n in classes
        # ))

        model.addConstrs((
            a_cap.sum(n, "*") <= 1
            for n in branch_nodes
        ))

        model.addConstrs((
            a_cap[n, i] >= a[n, i]
            for n in branch_nodes
            for i in range(n_features)
        ))

        model.addConstrs((
            a_cap[n, i] >= -a[n, i]
            for n in branch_nodes
            for i in range(n_features)
        ))

        return model

    def _set_gurobi_params(self):
        """Set Gurobi parameters."""
        model = self.model_
        model.Params.LogToConsole = self.verbose
        model.Params.OutputFlag = self.verbose
        if self.time_limit is not None:
            model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = 0.0
        # Need to reduce desired gap (exactly 1) by a small amount due to numerical issues
        model.Params.MIPGapAbs = 0.999

    
    def cuts(self):
        """Gurobi callback. Adds lazy cuts."""
        model = self.model_

        c, a, a_cap, b, g = model._vars

        X, y = model._X_y
        branch_nodes, leaf_nodes, nodes, arcs = model._flow_graph
        left_ancestors, right_ancestors = model._ancestors
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        lam, budget = model._cost
        epsilon = model._epsilon

        stop = True

        while(True):
            a_vals = model.getAttr('X', a)
            b_vals = model.getAttr('X', b)
            c_vals = model.getAttr('X', c)
            
            mu_i_xi = []  # List of tuples (mu, i, xi), one for every sample
            for i, x in enumerate(X):
                best_mu = float('inf')
                best_xi = {}
                for f in range(n_features):
                    best_xi[f] = 0.0
                
                t = 1
                while t <= len(branch_nodes):
                    if np.dot([a_vals[t, f] for f in range(n_features)], x) > b_vals[t]:
                        t = 2*t + 1
                    else:
                        t = 2*t
                        
                if c_vals[t, y[i]] < 0.5:
                    mu_i_xi.append((0.0, i, best_xi))
                    continue
                    
                for t in leaf_nodes:
                    if c_vals[t, y[i]] > 0.5:
                        continue

                    sub_model = gp.Model()
                    perturb_var = sub_model.addVars(n_features, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    perturb_cap_var = sub_model.addVars(n_features, lb=0, ub=GRB.INFINITY)

                    obj_fn = gp.quicksum(lam[f]*perturb_cap_var[f] for f in range(n_features))
                    sub_model.setObjective(obj_fn, GRB.MINIMIZE)

                    sub_model.addConstrs((
                        perturb_cap_var[n] >= perturb_var[n]
                        for n in range(n_features)
                    ))

                    sub_model.addConstrs((
                        perturb_cap_var[n] >= -perturb_var[n]
                        for n in range(n_features)
                    ))

                    sub_model.addConstrs((
                        gp.quicksum(a_vals[ancestor, f]*perturb_var[f] for f in range(n_features)) <= b_vals[ancestor] - gp.quicksum(a_vals[ancestor, f]*x[f] for f in range(n_features))
                        for ancestor in left_ancestors[t]
                    ))
                    sub_model.addConstrs((
                        gp.quicksum(a_vals[ancestor, f]*perturb_var[f] for f in range(n_features)) >= b_vals[ancestor] + epsilon - gp.quicksum(a_vals[ancestor, f]*x[f] for f in range(n_features))
                        for ancestor in right_ancestors[t]
                    ))
                    
                    sub_model.Params.LogToConsole = False

                    sub_model.optimize()

                    
                    if sub_model.status == 2 and sub_model.objVal < best_mu:
                        best_mu = sub_model.objVal
                        best_xi = sub_model.getAttr('X', perturb_var)

                mu_i_xi.append((best_mu, i, best_xi))

            perturb_set = []
            total_cost = 0
            for mu, i, xi in sorted(mu_i_xi):
                if total_cost + mu <= budget:
                    total_cost += mu
                    perturb_set.append([i, xi])
                else:
                    best_xi = {}
                    for f in range(n_features):
                        best_xi[f] = 0.0
                    perturb_set.append([i, best_xi])

            # if len(perturb_set) == 0:
            #     for mu, i, xi in sorted(mu_i_xi):
            #         temp = {}
            #         for f in range(n_features):
            #             temp[f] = 0.0
            #         perturb_set.append([i, temp])

            print("Perturbation: ", mu_i_xi)
            print("Selected data points: ", perturb_set)
            # print(a_vals)
            # print(b_vals)
            # print(c_vals)

            acc_count = 0
            for x_perturb in perturb_set:
                x = np.copy(X[x_perturb[0]])
                for j in x_perturb[1].keys():
                    x[j] += x_perturb[1][j]
                t = 1
                while t <= len(branch_nodes):
                    if np.dot([a_vals[t, f] for f in range(n_features)], x) > b_vals[t]:
                        t = 2*t + 1
                    else:
                        t = 2*t
                        
                if c_vals[t, y[x_perturb[0]]] > 0.5:
                    acc_count += 1

            if model.objVal <= acc_count:
                acc_count = 0
                for i, x in enumerate(X):
                    t = 1
                    while t <= len(branch_nodes):
                        if np.dot([a_vals[t, f] for f in range(n_features)], x) > b_vals[t]:
                            t = 2*t + 1
                        else:
                            t = 2*t
                            
                    if c_vals[t, y[i]] > 0.5:
                        acc_count += 1

                # stop = (model.objVal > acc_count)

                print("a: ", a_vals)
                print("b: ", b_vals)
                print("c: ", c_vals)
                print("Accuracy: ", acc_count/len(X))
                print("Objective Value: ", model.objVal)
                print("---------------------------------------------------------")
                print()
                break
            else:
                benders_cut_rhs = gp.LinExpr()
                for x_perturb in perturb_set:
                    x = np.copy(X[x_perturb[0]])
                    for j in x_perturb[1].keys():
                        x[j] += x_perturb[1][j]
                    t = 1
                    while t <= len(branch_nodes):
                        gamma = model.addVars([1], vtype=GRB.BINARY)
                        if np.dot([a_vals[t, f] for f in range(n_features)], x) > b_vals[t]:
                            model.addConstr((gamma[1] == 1) >> (np.dot([a[t, f] for f in range(n_features)], x) <= b[t]))
                            model.addConstr((gamma[1] == 0) >> (np.dot([a[t, f] for f in range(n_features)], x) >= b[t] + epsilon))
                            benders_cut_rhs.add(gamma[1])
                            t = 2*t + 1
                        else:
                            model.addConstr((gamma[1] == 1) >> (np.dot([a[t, f] for f in range(n_features)], x) >= b[t] + epsilon))
                            model.addConstr((gamma[1] == 0) >> (np.dot([a[t, f] for f in range(n_features)], x) <= b[t]))
                            benders_cut_rhs.add(gamma[1])
                            t = 2*t
                    # print(c[t, y[x_perturb[0]]])
                    benders_cut_rhs.add(c[t, y[x_perturb[0]]])

                model.addConstr(g.sum() <= benders_cut_rhs)
                # print(benders_cut_rhs)
                model.optimize()

                a_vals = model.getAttr('X', a)
                b_vals = model.getAttr('X', b)
                c_vals = model.getAttr('X', c)

                acc_count = 0
                for i, x in enumerate(X):
                    t = 1
                    while t <= len(branch_nodes):
                        if np.dot([a_vals[t, f] for f in range(n_features)], x) > b_vals[t]:
                            t = 2*t + 1
                        else:
                            t = 2*t
                            
                    if c_vals[t, y[i]] > 0.5:
                        acc_count += 1

                # stop = (model.objVal > acc_count)

                print("a: ", a_vals)
                print("b: ", b_vals)
                print("c: ", c_vals)
                print("Accuracy: ", acc_count/len(X))
                print("Objective Value: ", model.objVal)
                print("---------------------------------------------------------")
                print()
            