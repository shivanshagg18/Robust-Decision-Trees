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

class RDT(ClassifierMixin, BaseEstimator):
    def __init__(self, verbose=0, epsilon=0.01, max_depth=1):
        self.max_depth = max_depth
        self.verbose = verbose
        self.epsilon = epsilon

    def fit(self, X, y):
        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        
        self.first = self.first_model(X, y)
        self.set_gurobi_params(self.first)
        self.first.optimize()

        self.second = self.second_model(self.first)
        self.set_gurobi_params(self.second)
        self.second.optimize()
        
        return self
    
    def first_model(self, X, y):
        n_samples, n_features = X.shape
        classes = unique_labels(y)
        n_index = list(range(1, n_samples+1))
        
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
        left_nodes = set()
        right_nodes = set()
        for t in leaf_nodes:
            la = []
            ra = []
            curr_node = t
            while curr_node > 1:
                parent = int(curr_node/2)
                if curr_node == 2*parent:
                    la.append(parent)
                    left_nodes.add(curr_node)
                else:
                    ra.append(parent)
                    right_nodes.add(curr_node)
                curr_node = parent
            left_ancestors[t] = la
            right_ancestors[t] = ra
        
        model = gp.Model()
        model._X_y_n = X, y, n_index
        model._flow_graph = branch_nodes, leaf_nodes, nodes, arcs
        model._ancestors = left_ancestors, right_ancestors, left_nodes, right_nodes

        c = model.addVars(leaf_nodes, classes, vtype=GRB.BINARY, name="c")
        a = model.addVars(branch_nodes, n_features, lb=-1, ub=1, name="a")
        a_cap = model.addVars(branch_nodes, n_features, lb=0, ub=1, name="a_cap")
        b = model.addVars(branch_nodes, lb=-1, ub=1, name="b")
        b_cap = model.addVars(branch_nodes, lb=0, ub=1, name="b_cap")
        w = model.addVars(n_index, nodes, vtype=GRB.BINARY, name="w")
        z = model.addVars(n_index, leaf_nodes, vtype=GRB.BINARY, name="z")
        gamma = model.addVars(n_index, nodes, vtype=GRB.BINARY, name="z")

        model._vars = c, a, a_cap, b, b_cap, w, z, gamma

        model._epsilon = self.epsilon

        obj_fn = z.sum()
        model.setObjective(obj_fn, GRB.MAXIMIZE)

        model.addConstrs((
            c.sum(n, "*") == 1
            for n in leaf_nodes
        ))

        model.addConstrs((
            a_cap.sum(n, "*") <= 1
            for n in branch_nodes
        ))

        model.addConstrs((
            a_cap[n, i] == gp.abs_(a[n, i])
            for n in branch_nodes
            for i in range(n_features)
        ))

        model.addConstrs((
            b_cap[n] == gp.abs_(b[n])
            for n in branch_nodes
        ))

        model.addConstrs((
            b_cap[n] >= self.epsilon + (self.epsilon/10)
            for n in branch_nodes
        ))

        model.addConstrs((
            w[i, n] == w[i, 2*n] + w[i, 2*n + 1]
            for i in n_index
            for n in branch_nodes
        ))

        model.addConstrs((
            w[i, n] == z[i, n]
            for i in n_index
            for n in leaf_nodes
        ))

        model.addConstrs((
            z[i, n] <= c[n, y[i-1]]
            for i in n_index
            for n in leaf_nodes
        ))

        model.addConstrs((
            w[i, n] <= gamma[i, n]
            for i in n_index
            for n in nodes
        ))

        model.addConstrs((
            gamma[i, 1] == 1
            for i in n_index
        ))

        model.addConstrs((
            (gamma[i, n] == 1) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1]) <= b[n//2])
            for i in n_index
            for n in left_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 0) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1]) >= b[n//2] + self.epsilon)
            for i in n_index
            for n in left_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 0) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1]) <= b[n//2])
            for i in n_index
            for n in right_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 1) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1]) >= b[n//2] + self.epsilon)
            for i in n_index
            for n in right_nodes
        ))

        return model
    

    def second_model(self, first_model):
        X, y_class, n_index = first_model._X_y_n
        branch_nodes, leaf_nodes, nodes, arcs = first_model._flow_graph
        left_ancestors, right_ancestors, left_nodes, right_nodes = first_model._ancestors
        c, a, a_cap, b, b_cap, w, z, gamma = first_model._vars

        gamma_vals = first_model.getAttr('X', gamma)
        c_vals = first_model.getAttr('X', c)

        model = gp.Model()
        y = model.addVars(n_index, nodes, vtype=GRB.BINARY, name="y")
        p = model.addVars(n_index, nodes, vtype=GRB.BINARY, name="p")
        q = model.addVars(n_index, leaf_nodes, vtype=GRB.BINARY, name="q")

        model._vars = y, p, q

        obj_fn = gp.LinExpr()
        for i in n_index:
            for n in leaf_nodes:
                obj_fn.add(c_vals[n, y_class[i-1]]*q[i, n])

            for n in nodes:
                obj_fn.add(gamma_vals[i, n]*p[i, n])

        model.setObjective(obj_fn, GRB.MINIMIZE)

        model.addConstrs((
            y[i, n//2] - y[i, n] + p[i, n] >= 0
            for i in n_index
            for n in left_nodes
        ))

        model.addConstrs((
            y[i, n//2] - y[i, n] + p[i, n] >= 0
            for i in n_index
            for n in right_nodes
        ))

        model.addConstrs((
            -y[i, 1] + p[i, 1] >= 0
            for i in n_index
        ))

        model.addConstrs((
            y[i, n] + q[i, n] >= 1
            for i in n_index
            for n in leaf_nodes
        ))

        return model


    def set_gurobi_params(self, model):
        model.Params.LogToConsole = True
        model.Params.OutputFlag = False
        model.Params.MIPGap = 0.0
        model.Params.MIPGapAbs = 0.999