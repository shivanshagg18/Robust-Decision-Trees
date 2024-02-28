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
    def __init__(self, verbose=0, epsilon=0.01, max_depth=1, budget=0, num_cuts=5):
        self.max_depth = max_depth
        self.verbose = verbose
        self.epsilon = epsilon
        self.budget = budget
        self.num_cuts = num_cuts

    def fit(self, X, y):
        # Check that X and y have correct shape, convert X and y to ndarrays
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        classes = unique_labels(y)

        a_vals = []
        b_vals = []
        c_vals = []
        perturb_vals = np.zeros((self.num_cuts, n_samples, n_features))
        
        for cut in range(self.num_cuts):
            self.first = self.first_model(X, y, cut)
            self.set_gurobi_params(self.first)
            self.first.optimize()
            c, a, a_cap, b, b_cap, w, z, gamma, perturb = self.first._vars
            a_vals.append(self.first.getAttr('X', a))
            b_vals.append(self.first.getAttr('X', b))
            c_vals.append(self.first.getAttr('X', c))
            perturb_vals[cut] = perturb

        # self.second = self.second_model(self.first)
        # self.set_gurobi_params(self.second)
        # self.second.optimize()

        # self.third = self.third_model(self.first)
        # self.set_gurobi_params(self.third)
        # self.third.optimize()

        # self.cuts()
        
        return self
    
    def first_model(self, X, y, cut):
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

        perturb = np.zeros((n_samples, n_features))

        model._vars = c, a, a_cap, b, b_cap, w, z, gamma, perturb

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

        for i in range(n_samples):
            if cut != 0:
                perturb_temp = np.random.dirichlet(np.ones(n_features),size=1)[0]*self.budget/n_samples
                perturb[i] = perturb_temp
            else:
                perturb_temp = np.zeros((n_features))
                perturb[i] = perturb_temp 

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
            (gamma[i, n] == 1) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1] + perturb[i-1]) <= b[n//2] -  self.epsilon)
            for i in n_index
            for n in left_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 0) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1] + perturb[i-1]) >= b[n//2] + self.epsilon)
            for i in n_index
            for n in left_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 0) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1] + perturb[i-1]) <= b[n//2] -  self.epsilon)
            for i in n_index
            for n in right_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 1) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1] + perturb[i-1]) >= b[n//2] + self.epsilon)
            for i in n_index
            for n in right_nodes
        ))

        return model
    

    # def second_model(self, first_model):
    #     X, y_class, n_index = first_model._X_y_n
    #     branch_nodes, leaf_nodes, nodes, arcs = first_model._flow_graph
    #     left_ancestors, right_ancestors, left_nodes, right_nodes = first_model._ancestors
    #     c, a, a_cap, b, b_cap, w, z, gamma = first_model._vars

    #     gamma_vals = first_model.getAttr('X', gamma)
    #     c_vals = first_model.getAttr('X', c)

    #     model = gp.Model()
    #     y = model.addVars(n_index, nodes, vtype=GRB.BINARY, name="y")
    #     p = model.addVars(n_index, nodes, vtype=GRB.BINARY, name="p")
    #     q = model.addVars(n_index, leaf_nodes, vtype=GRB.BINARY, name="q")

    #     model._vars = y, p, q

    #     obj_fn = gp.LinExpr()
    #     for i in n_index:
    #         for n in leaf_nodes:
    #             obj_fn.add(c_vals[n, y_class[i-1]]*q[i, n])

    #         for n in nodes:
    #             obj_fn.add(gamma_vals[i, n]*p[i, n])

    #     model.setObjective(obj_fn, GRB.MINIMIZE)

    #     model.addConstrs((
    #         y[i, n//2] - y[i, n] + p[i, n] >= 0
    #         for i in n_index
    #         for n in left_nodes
    #     ))

    #     model.addConstrs((
    #         y[i, n//2] - y[i, n] + p[i, n] >= 0
    #         for i in n_index
    #         for n in right_nodes
    #     ))

    #     model.addConstrs((
    #         -y[i, 1] + p[i, 1] >= 0
    #         for i in n_index
    #     ))

    #     model.addConstrs((
    #         y[i, n] + q[i, n] >= 1
    #         for i in n_index
    #         for n in leaf_nodes
    #     ))

    #     return model
    
    def third_model(self, first_model):
        X, y_class, n_index = first_model._X_y_n
        branch_nodes, leaf_nodes, nodes, arcs = first_model._flow_graph
        left_ancestors, right_ancestors, left_nodes, right_nodes = first_model._ancestors
        n_samples, n_features = X.shape
        classes = unique_labels(y_class)

        p_vals = np.zeros((self.num_cuts, n_samples+1, len(nodes)+1))
        q_vals = np.zeros((self.num_cuts, n_samples+1, len(nodes)+1))

        c_vals, a_vals, _a_cap, b_vals, _b_cap, _w, _z, _gamma, perturb = first_model._vars

        a_vals = first_model.getAttr('X', a_vals)
        b_vals = first_model.getAttr('X', b_vals)
        c_vals = first_model.getAttr('X', c_vals)
        gamma_vals = first_model.getAttr('X', _gamma)

        model = gp.Model()

        c = model.addVars(leaf_nodes, classes, vtype=GRB.BINARY, name="c")
        a = model.addVars(branch_nodes, n_features, lb=-1, ub=1, name="a")
        a_cap = model.addVars(branch_nodes, n_features, lb=0, ub=1, name="a_cap")
        b = model.addVars(branch_nodes, lb=-1, ub=1, name="b")
        b_cap = model.addVars(branch_nodes, lb=0, ub=1, name="b_cap")
        gamma = model.addVars(self.num_cuts, n_index, nodes, vtype=GRB.BINARY, name="z")

        g = model.addVars(n_samples, vtype=GRB.BINARY, ub=1, name="t")

        for n in branch_nodes:
            b[n].Start = b_vals[n]
            b_cap[n].Start = np.abs(b_vals[n])

            for f in range(n_features):
                a[n, f].Start = a_vals[n, f]
                a_cap[n, f].Start = np.abs(a_vals[n, f])

        for n in leaf_nodes:
            for cls in classes:
                c[n, cls].Start = c_vals[n, cls]


        for j in range(self.num_cuts):
            for i, x in enumerate(X):
                t = 1
                while t <= len(branch_nodes):
                    if (np.dot([a_vals[t, f] for f in range(n_features)], x + perturb[j, i]) > b_vals[t] + self.epsilon) or (np.abs(np.dot([a_vals[t, f] for f in range(n_features)], x + perturb[j, i]) - b_vals[t] - self.epsilon) <= 0.0001):
                        p_vals[j, i+1, 2*t] = 1
                        t = 2*t + 1
                    else:
                        p_vals[j, i+1, 2*t + 1] = 1
                        t = 2*t
                        
                if c_vals[t, y_class[i]] > 0.5:
                    q_vals[j, i+1, t] = 1

                    for n in nodes:
                        gamma[j, i+1, n].Start = gamma_vals[j, i+1, n] 

        model._vars = c, a, a_cap, b, b_cap, gamma, g

        lam = np.array([1 for i in range(n_features)])
        budget = self.budget
        model._cost = lam, budget
        model._epsilon = self.epsilon

        obj_fn = g.sum()
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

        for j in range(self.num_cuts):
            model.addConstrs((
                gamma[j, i, 1] == 1
                for i in n_index
            ))

            model.addConstrs((
                (gamma[j, i, n] == 1) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1] + perturb[j, i-1]) <= b[n//2] -  self.epsilon)
                for i in n_index
                for n in left_nodes
            ))

            model.addConstrs((
                (gamma[j, i, n] == 0) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1] + perturb[j, i-1]) >= b[n//2] + self.epsilon)
                for i in n_index
                for n in left_nodes
            ))

            model.addConstrs((
                (gamma[j, i, n] == 0) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1] + perturb[j, i-1]) <= b[n//2] -  self.epsilon)
                for i in n_index
                for n in right_nodes
            ))

            model.addConstrs((
                (gamma[j, i, n] == 1) >> (np.dot([a[n//2, f] for f in range(n_features)], X[i-1] + perturb[j, i-1]) >= b[n//2] + self.epsilon)
                for i in n_index
                for n in right_nodes
            ))

            benders_cut_rhs = gp.LinExpr()
            for i in n_index:
                for n in leaf_nodes:
                    benders_cut_rhs.add(c[n, y_class[i-1]]*q_vals[j, i, n])

                for n in nodes:
                    benders_cut_rhs.add(gamma[j, i, n]*p_vals[j, i, n])

            model.addConstr(g.sum() <= benders_cut_rhs)

        return model
    
    def cuts(self):
        model = self.third

        c, a, a_cap, b, b_cap, _gamma, g = model._vars

        X, y_class, n_index = self.first._X_y_n
        branch_nodes, leaf_nodes, nodes, arcs = self.first._flow_graph
        left_ancestors, right_ancestors, left_nodes, right_nodes = self.first._ancestors
        n_samples, n_features = X.shape
        classes = unique_labels(y_class)
        lam, budget = model._cost
        epsilon = model._epsilon
        gamma_con = 0

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
                    if (np.dot([a_vals[t, f] for f in range(n_features)], x) > b_vals[t] + self.epsilon) or (np.abs(np.dot([a_vals[t, f] for f in range(n_features)], x) - b_vals[t] - self.epsilon) <= 0.0001):
                        t = 2*t + 1
                    else:
                        t = 2*t
                        
                if c_vals[t, y_class[i]] < 0.5:
                    mu_i_xi.append((0.0, i, best_xi))
                    continue
                    
                for t in leaf_nodes:
                    if c_vals[t, y_class[i]] > 0.5:
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
                        gp.quicksum(a_vals[ancestor, f]*perturb_var[f] for f in range(n_features)) <= b_vals[ancestor] - 2*epsilon - gp.quicksum(a_vals[ancestor, f]*x[f] for f in range(n_features))
                        for ancestor in left_ancestors[t]
                    ))

                    sub_model.addConstrs((
                        gp.quicksum(a_vals[ancestor, f]*perturb_var[f] for f in range(n_features)) >= b_vals[ancestor] + 2*epsilon - gp.quicksum(a_vals[ancestor, f]*x[f] for f in range(n_features))
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

            if self.verbose:
                print("Perturbation:")
                for per in mu_i_xi:
                    print("Data point " + str(per[1]) + ": " + str(per[2]))

                print()
                print("Selected data points:")
                for per in perturb_set:
                    print("Data point " + str(per[0]) + ": " + str(per[1]))


            acc_count = 0
            for x_perturb in perturb_set:
                x = np.copy(X[x_perturb[0]])
                for j in x_perturb[1].keys():
                    x[j] += x_perturb[1][j]
                t = 1
                while t <= len(branch_nodes):
                    if (np.dot([a_vals[t, f] for f in range(n_features)], x) > b_vals[t] + self.epsilon) or (np.abs(np.dot([a_vals[t, f] for f in range(n_features)], x) - b_vals[t] - self.epsilon) <= 0.0001):
                        t = 2*t + 1
                    else:
                        t = 2*t
                        
                if c_vals[t, y_class[x_perturb[0]]] > 0.5:
                    acc_count += 1

            print("Accuracy (perturbed) before adding cut", acc_count/len(X))

            if model.objVal > acc_count:
                if self.verbose:
                    print()
                    print("Constraints added:")
                benders_cut_rhs = gp.LinExpr()
                temp = "g <= "
                for x_perturb in perturb_set:
                    x = np.copy(X[x_perturb[0]])
                    for j in x_perturb[1].keys():
                        x[j] += x_perturb[1][j]
                    t = 1
                    while t <= len(branch_nodes):
                        gamma = model.addVars([1], vtype=GRB.BINARY)
                        gamma_con += 1
                        temp += "gamma_" + str(gamma_con) + " + "
                        if np.dot([a_vals[t, f] for f in range(n_features)], x) >= b_vals[t] + epsilon:
                            temp_2 = "gamma_" + str(gamma_con) + " == 1 >> "
                            for f in range(n_features):
                                temp_2 = temp_2 + str("a[" + str(t) + "," + str(f) + "]*" + str(x[f]) + " + ")

                            temp_2 = temp_2[:-3] + " <= " + "b[" + str(t) + "] - epsilon"
                            if self.verbose:
                                print(temp_2)
                            
                            model.addConstr((gamma[1] == 1) >> (np.dot([a[t, f] for f in range(n_features)], x) <= b[t] - epsilon))
                            model.addConstr((gamma[1] == 0) >> (np.dot([a[t, f] for f in range(n_features)], x) >= b[t] + epsilon))
                            t = 2*t + 1
                        else:
                            temp_2 = "gamma_" + str(gamma_con) + " == 1 >> "
                            for f in range(n_features):
                                temp_2 = temp_2 + str("a[" + str(t) + "," + str(f) + "]*" + str(x[f]) + " + ")

                            temp_2 = temp_2[:-3] + " >= " + "b[" + str(t) + "] +  epsilon"
                            if self.verbose:
                                print(temp_2)
                            
                            model.addConstr((gamma[1] == 1) >> (np.dot([a[t, f] for f in range(n_features)], x) >= b[t] + epsilon))
                            model.addConstr((gamma[1] == 0) >> (np.dot([a[t, f] for f in range(n_features)], x) <= b[t] - epsilon))
                            t = 2*t
                        
                        benders_cut_rhs.add(gamma[1])
                    benders_cut_rhs.add(c[t, y_class[x_perturb[0]]])
                    temp = temp + "c[" + str(t) + "," + str(y_class[x_perturb[0]]) + "] + "

                if self.verbose:
                    print(temp[-3:])
                
                model.addConstr(g.sum() <= benders_cut_rhs)
                model.optimize()
            else:
                break

            a_vals = model.getAttr('X', a)
            b_vals = model.getAttr('X', b)
            c_vals = model.getAttr('X', c)

            acc_count = 0
            for i, x in enumerate(X):
                t = 1
                while t <= len(branch_nodes):
                    if (np.dot([a_vals[t, f] for f in range(n_features)], x) > b_vals[t] + self.epsilon) or (np.abs(np.dot([a_vals[t, f] for f in range(n_features)], x) - b_vals[t] - self.epsilon) <= 0.0001):
                        t = 2*t + 1
                    else:
                        t = 2*t
                        
                if c_vals[t, y_class[i]] > 0.5:
                    acc_count += 1

            if self.verbose:
                print()
                print("Hyperplanes:")
                for n in branch_nodes:
                    temp = str(n) + ": "
                    for i in range(n_features):
                        temp = temp + str(a_vals[n, i]) + " + "
                    temp = temp[:-3] + " <= " + str(b_vals[n])
                    print(temp)

                print()
                print("Leaf nodes:")
                for n in leaf_nodes:
                    temp = str(n) + ": "
                    for i in classes:
                        if c_vals[n, i] == 1:
                            temp = temp + str(i)
                    print(temp)

                print()
            
            print("Accuracy: ", acc_count/len(X))
            print("Objective Value: ", model.objVal)
            print("---------------------------------------------------------")
            print()


    def set_gurobi_params(self, model):
        model.Params.LogToConsole = False
        model.Params.OutputFlag = False
        # model.Params.MIPGap = 0.0
        # model.Params.MIPGapAbs = 0.999