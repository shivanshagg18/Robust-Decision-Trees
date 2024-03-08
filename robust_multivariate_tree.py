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
    def __init__(self, verbose=0, epsilon=0.01, max_depth=1, budget=0, num_cuts=1, time_limit=20, obj_relax=10):
        self.max_depth = max_depth
        self.verbose = verbose
        self.epsilon = epsilon
        self.budget = budget
        self.num_cuts = num_cuts
        self.time_limit = time_limit
        self.obj_relax = obj_relax
        self.time_stats = pd.DataFrame({"Perturbed Accuracy": [], "Accuracy": [], "Objective Value": [], "Time": []})

    def set_gurobi_params(self, model):
        model.Params.LogToConsole = False
        model.Params.OutputFlag = False
        model.Params.TimeLimit = self.time_limit

    def define_variables(self, X, y):
        self.X, self.y = check_X_y(X, y)
        
        self.n_samples, self.n_features = self.X.shape
        self.classes = unique_labels(self.y)
        self.n_index = list(range(1, self.n_samples+1))
        
        self.branch_nodes = list(range(1, 2**self.max_depth))
        self.leaf_nodes = list(range(2**self.max_depth, 2**(self.max_depth+1)))
        self.nodes = list(range(1, 2**(self.max_depth+1)))
        self.arcs = (
            [('s', 1)]
            + [(n, 2*n) for n in self.branch_nodes]
            + [(n, 2*n+1) for n in self.branch_nodes]
            + [(n, 't') for n in self.branch_nodes+self.leaf_nodes]
        )

        self.left_ancestors = {}
        self.right_ancestors = {}
        self.left_nodes = set()
        self.right_nodes = set()
        for t in self.leaf_nodes:
            la = []
            ra = []
            curr_node = t
            while curr_node > 1:
                parent = int(curr_node/2)
                if curr_node == 2*parent:
                    la.append(parent)
                    self.left_nodes.add(curr_node)
                else:
                    ra.append(parent)
                    self.right_nodes.add(curr_node)
                curr_node = parent
            self.left_ancestors[t] = la
            self.right_ancestors[t] = ra

    def fit(self, X, y):
        self.define_variables(X, y)

        a_vals = []
        b_vals = []
        c_vals = []
        gamma_vals = []
        perturb_vals = np.zeros((self.num_cuts, self.n_samples, self.n_features))
        
        for cut in range(self.num_cuts):
            self.first = self.first_model(self.X, self.y, cut)
            self.set_gurobi_params(self.first)
            self.first.optimize()
            c, a, a_cap, b, b_cap, w, z, gamma, perturb = self.first._vars
            a_vals.append(self.first.getAttr('X', a))
            b_vals.append(self.first.getAttr('X', b))
            c_vals.append(self.first.getAttr('X', c))
            gamma_vals.append(self.first.getAttr('X', gamma))
            perturb_vals[cut] = perturb

        self.second = self.second_model(a_vals, b_vals, c_vals, gamma_vals, perturb_vals)
        self.set_gurobi_params(self.second)
        self.second.optimize()

        a_vals = self.second.getAttr('X', a)
        b_vals = self.second.getAttr('X', b)
        c_vals = self.second.getAttr('X', c)

        acc_count = 0
        for i, x in enumerate(self.X):
            t = 1
            while t <= len(self.branch_nodes):
                if (np.dot([a_vals[t, f] for f in range(self.n_features)], x) > b_vals[t] + self.epsilon) or (np.abs(np.dot([a_vals[t, f] for f in range(self.n_features)], x) - b_vals[t] - self.epsilon) <= self.epsilon/100):
                    t = 2*t + 1
                else:
                    t = 2*t
                    
            if c_vals[t, self.y[i]] > 0.5:
                acc_count += 1

        if self.verbose:
            print()
            print("Hyperplanes:")
            for n in self.branch_nodes:
                temp = str(n) + ": "
                for i in range(self.n_features):
                    temp = temp + str(a_vals[n, i]) + "x" + str(i) + " + "
                temp = temp[:-3] + " <= " + str(b_vals[n])
                print(temp)

            print()
            print("Leaf nodes:")
            for n in self.leaf_nodes:
                temp = str(n) + ": "
                for i in self.classes:
                    if c_vals[n, i] == 1:
                        temp = temp + str(i)
                print(temp)

            print()
        
        print("Accuracy: ", acc_count/len(self.X))
        print("Objective Value: ", self.second.objVal)
        print("---------------------------------------------------------")

        self.cuts()
        
        return self
    
    def first_model(self, X, y, cut):
        model = gp.Model()

        c = model.addVars(self.leaf_nodes, self.classes, vtype=GRB.BINARY, name="c")
        a = model.addVars(self.branch_nodes, self.n_features, lb=-1, ub=1, name="a")
        a_cap = model.addVars(self.branch_nodes, self.n_features, lb=0, ub=1, name="a_cap")
        b = model.addVars(self.branch_nodes, lb=-1, ub=1, name="b")
        b_cap = model.addVars(self.branch_nodes, lb=0, ub=1, name="b_cap")
        w = model.addVars(self.n_index, self.nodes, vtype=GRB.BINARY, name="w")
        z = model.addVars(self.n_index, self.leaf_nodes, vtype=GRB.BINARY, name="z")
        gamma = model.addVars(self.n_index, self.nodes, vtype=GRB.BINARY, name="z")

        perturb = np.zeros((self.n_samples, self.n_features))

        for i in range(self.n_samples):
            if cut != 0:
                perturb[i] = np.random.dirichlet(np.ones(self.n_features),size=1)[0]*self.budget/self.n_samples

            for f in range(self.n_features):
                rndm = random.randint(0, 1)
                if rndm == 0:
                    perturb[i, f] = -perturb[i, f]

        model._vars = c, a, a_cap, b, b_cap, w, z, gamma, perturb

        obj_fn = z.sum()
        model.setObjective(obj_fn, GRB.MAXIMIZE)

        model.addConstrs((
            c.sum(n, "*") == 1
            for n in self.leaf_nodes
        ))

        model.addConstrs((
            a_cap.sum(n, "*") <= 1
            for n in self.branch_nodes
        ))

        model.addConstrs((
            a_cap[n, i] == gp.abs_(a[n, i])
            for n in self.branch_nodes
            for i in range(self.n_features)
        ))

        model.addConstrs((
            b_cap[n] == gp.abs_(b[n])
            for n in self.branch_nodes
        ))

        model.addConstrs((
            w[i, n] == w[i, 2*n] + w[i, 2*n + 1]
            for i in self.n_index
            for n in self.branch_nodes
        ))

        model.addConstrs((
            w[i, n] == z[i, n]
            for i in self.n_index
            for n in self.leaf_nodes
        ))

        model.addConstrs((
            z[i, n] <= c[n, y[i-1]]
            for i in self.n_index
            for n in self.leaf_nodes
        ))

        model.addConstrs((
            w[i, n] <= gamma[i, n]
            for i in self.n_index
            for n in self.nodes
        ))

        model.addConstrs((
            gamma[i, 1] == 1
            for i in self.n_index
        ))

        model.addConstrs((
            (gamma[i, n] == 1) >> (np.dot([a[n//2, f] for f in range(self.n_features)], X[i-1] + perturb[i-1]) <= b[n//2] -  self.epsilon)
            for i in self.n_index
            for n in self.left_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 0) >> (np.dot([a[n//2, f] for f in range(self.n_features)], X[i-1] + perturb[i-1]) >= b[n//2] + self.epsilon)
            for i in self.n_index
            for n in self.left_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 0) >> (np.dot([a[n//2, f] for f in range(self.n_features)], X[i-1] + perturb[i-1]) <= b[n//2] -  self.epsilon)
            for i in self.n_index
            for n in self.right_nodes
        ))

        model.addConstrs((
            (gamma[i, n] == 1) >> (np.dot([a[n//2, f] for f in range(self.n_features)], X[i-1] + perturb[i-1]) >= b[n//2] + self.epsilon)
            for i in self.n_index
            for n in self.right_nodes
        ))

        return model
    
    def second_model(self, a_vals, b_vals, c_vals, gamma_vals, perturb_vals):
        p_vals = np.zeros((self.num_cuts, self.n_samples+1, len(self.nodes)+1))
        q_vals = np.zeros((self.num_cuts, self.n_samples+1, len(self.nodes)+1))

        model = gp.Model()

        c = model.addVars(self.leaf_nodes, self.classes, vtype=GRB.BINARY, name="c")
        a = model.addVars(self.branch_nodes, self.n_features, lb=-1, ub=1, name="a")
        a_cap = model.addVars(self.branch_nodes, self.n_features, lb=0, ub=1, name="a_cap")
        b = model.addVars(self.branch_nodes, lb=-1, ub=1, name="b")
        b_cap = model.addVars(self.branch_nodes, lb=0, ub=1, name="b_cap")
        gamma = model.addVars(self.num_cuts, self.n_index, self.nodes, vtype=GRB.BINARY, name="gamma")

        g = model.addVars(self.n_samples, vtype=GRB.BINARY, ub=1, name="t")

        for n in self.branch_nodes:
            b[n].Start = b_vals[0][n]
            b_cap[n].Start = np.abs(b_vals[0][n])

            for f in range(self.n_features):
                a[n, f].Start = a_vals[0][n, f]
                a_cap[n, f].Start = np.abs(a_vals[0][n, f])

        for n in self.leaf_nodes:
            for cls in self.classes:
                c[n, cls].Start = c_vals[0][n, cls]


        for j in range(self.num_cuts):
            for i, x in enumerate(self.X):
                t = 1
                while t <= len(self.branch_nodes):
                    if (np.dot([a_vals[j][t, f] for f in range(self.n_features)], x + perturb_vals[j, i]) > b_vals[j][t] + self.epsilon) or (np.abs(np.dot([a_vals[j][t, f] for f in range(self.n_features)], x + perturb_vals[j, i]) - b_vals[j][t] - self.epsilon) <= self.epsilon/100):
                        p_vals[j, i+1, 2*t] = 1
                        t = 2*t + 1
                    else:
                        p_vals[j, i+1, 2*t + 1] = 1
                        t = 2*t
                        
                if c_vals[j][t, self.y[i]] > 0.5:
                    q_vals[j, i+1, t] = 1

                for n in self.nodes:
                    gamma[j, i+1, n].Start = gamma_vals[j][i+1, n]

        model._vars = c, a, a_cap, b, b_cap, gamma, g

        lam = np.array([1 for i in range(self.n_features)])
        budget = self.budget
        model._cost = lam, budget

        obj_fn = g.sum()
        model.setObjective(obj_fn, GRB.MAXIMIZE)

        model.addConstrs((
            c.sum(n, "*") == 1
            for n in self.leaf_nodes
        ))

        model.addConstrs((
            c.sum("*", n) >= 1
            for n in self.classes
        ))

        model.addConstrs((
            a_cap.sum(n, "*") <= 1
            for n in self.branch_nodes
        ))

        model.addConstrs((
            a_cap[n, i] == gp.abs_(a[n, i])
            for n in self.branch_nodes
            for i in range(self.n_features)
        ))

        model.addConstrs((
            b_cap[n] == gp.abs_(b[n])
            for n in self.branch_nodes
        ))

        for j in range(self.num_cuts):
            model.addConstrs((
                gamma[j, i, 1] == 1
                for i in self.n_index
            ))

            model.addConstrs((
                (gamma[j, i, n] == 1) >> (np.dot([a[n//2, f] for f in range(self.n_features)], self.X[i-1] + perturb_vals[j, i-1]) <= b[n//2] -  self.epsilon)
                for i in self.n_index
                for n in self.left_nodes
            ))

            model.addConstrs((
                (gamma[j, i, n] == 0) >> (np.dot([a[n//2, f] for f in range(self.n_features)], self.X[i-1] + perturb_vals[j, i-1]) >= b[n//2] + self.epsilon)
                for i in self.n_index
                for n in self.left_nodes
            ))

            model.addConstrs((
                (gamma[j, i, n] == 0) >> (np.dot([a[n//2, f] for f in range(self.n_features)], self.X[i-1] + perturb_vals[j, i-1]) <= b[n//2] -  self.epsilon)
                for i in self.n_index
                for n in self.right_nodes
            ))

            model.addConstrs((
                (gamma[j, i, n] == 1) >> (np.dot([a[n//2, f] for f in range(self.n_features)], self.X[i-1] + perturb_vals[j, i-1]) >= b[n//2] + self.epsilon)
                for i in self.n_index
                for n in self.right_nodes
            ))

            benders_cut_rhs = gp.LinExpr()
            for i in self.n_index:
                for n in self.leaf_nodes:
                    benders_cut_rhs.add(c[n, self.y[i-1]]*q_vals[j, i, n])

                for n in self.nodes:
                    benders_cut_rhs.add(gamma[j, i, n]*p_vals[j, i, n])

            model.addConstr(g.sum() <= benders_cut_rhs)

        return model
    
    def cuts(self):
        model = self.second

        c, a, a_cap, b, b_cap, _gamma, g = model._vars
        lam, budget = model._cost

        gamma_con = 0
        iter_count = 0

        while(True):
            iter_count += 1
            print("Iteration Number:", iter_count)
            print()

            a_vals = model.getAttr('X', a)
            b_vals = model.getAttr('X', b)
            c_vals = model.getAttr('X', c)
            
            mu_i_xi = []
            for i, x in enumerate(self.X):
                best_mu = float('inf')
                best_xi = {}
                for f in range(self.n_features):
                    best_xi[f] = 0.0
                
                t = 1
                while t <= len(self.branch_nodes):
                    if (np.dot([a_vals[t, f] for f in range(self.n_features)], x) > b_vals[t] + self.epsilon) or (np.abs(np.dot([a_vals[t, f] for f in range(self.n_features)], x) - b_vals[t] - self.epsilon) <= self.epsilon/100):
                        t = 2*t + 1
                    else:
                        t = 2*t
                        
                if c_vals[t, self.y[i]] < 0.5:
                    mu_i_xi.append((0.0, i, best_xi))
                    continue
                    
                for t in self.leaf_nodes:
                    if c_vals[t, self.y[i]] > 0.5:
                        continue

                    sub_model = gp.Model()
                    self.set_gurobi_params(sub_model)
                    perturb_var = sub_model.addVars(self.n_features, lb=-GRB.INFINITY, ub=GRB.INFINITY)
                    perturb_cap_var = sub_model.addVars(self.n_features, lb=0, ub=GRB.INFINITY)

                    obj_fn = gp.quicksum(lam[f]*perturb_cap_var[f] for f in range(self.n_features))
                    sub_model.setObjective(obj_fn, GRB.MINIMIZE)

                    sub_model.addConstrs((
                        perturb_cap_var[n] >= perturb_var[n]
                        for n in range(self.n_features)
                    ))

                    sub_model.addConstrs((
                        perturb_cap_var[n] >= -perturb_var[n]
                        for n in range(self.n_features)
                    ))

                    sub_model.addConstrs((
                        perturb_cap_var[n] <= 0.05
                        for n in range(self.n_features)
                    ))

                    sub_model.addConstrs((
                        perturb_cap_var[n] >= 0.01
                        for n in range(self.n_features)
                    ))

                    sub_model.addConstrs((
                        x[n] + perturb_var[n] <= 1
                        for n in range(self.n_features)
                    ))

                    sub_model.addConstrs((
                        x[n] + perturb_var[n] >= 0
                        for n in range(self.n_features)
                    ))


                    sub_model.addConstrs((
                        gp.quicksum(a_vals[ancestor, f]*perturb_var[f] for f in range(self.n_features)) <= b_vals[ancestor] - 2*self.epsilon - gp.quicksum(a_vals[ancestor, f]*x[f] for f in range(self.n_features))
                        for ancestor in self.left_ancestors[t]
                    ))

                    sub_model.addConstrs((
                        gp.quicksum(a_vals[ancestor, f]*perturb_var[f] for f in range(self.n_features)) >= b_vals[ancestor] + 2*self.epsilon - gp.quicksum(a_vals[ancestor, f]*x[f] for f in range(self.n_features))
                        for ancestor in self.right_ancestors[t]
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
                    for f in range(self.n_features):
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
                x = np.copy(self.X[x_perturb[0]])
                for j in x_perturb[1].keys():
                    x[j] += x_perturb[1][j]
                t = 1
                while t <= len(self.branch_nodes):
                    if (np.dot([a_vals[t, f] for f in range(self.n_features)], x) > b_vals[t] + self.epsilon) or (np.abs(np.dot([a_vals[t, f] for f in range(self.n_features)], x) - b_vals[t] - self.epsilon) <= self.epsilon/100):
                        t = 2*t + 1
                    else:
                        t = 2*t
                        
                if c_vals[t, self.y[x_perturb[0]]] > 0.5:
                    acc_count += 1

            print("Accuracy (perturbed) before adding cut: ", acc_count/len(self.X))
            time_stats_row = [acc_count/len(self.X)]

            if model.objVal > acc_count + (iter_count//self.obj_relax):
                if self.verbose:
                    print()
                    print("Logical Constraints added:")
                benders_cut_rhs = gp.LinExpr()
                temp = "g <= "
                for x_perturb in perturb_set:
                    x = np.copy(self.X[x_perturb[0]])
                    for j in x_perturb[1].keys():
                        x[j] += x_perturb[1][j]
                    t = 1
                    while t <= len(self.branch_nodes):
                        gamma = model.addVars([1], vtype=GRB.BINARY)
                        gamma_con += 1
                        temp += "gamma_" + str(gamma_con) + " + "
                        if np.dot([a_vals[t, f] for f in range(self.n_features)], x) >= b_vals[t] + self.epsilon:
                            temp_2 = "gamma_" + str(gamma_con) + " == 1 >> "
                            for f in range(self.n_features):
                                temp_2 = temp_2 + str("a[" + str(t) + "," + str(f) + "]*" + str(x[f]) + " + ")

                            temp_2 = temp_2[:-3] + " <= " + "b[" + str(t) + "] - epsilon"
                            if self.verbose:
                                print(temp_2)
                            
                            model.addConstr((gamma[1] == 1) >> (np.dot([a[t, f] for f in range(self.n_features)], x) <= b[t] - self.epsilon))
                            model.addConstr((gamma[1] == 0) >> (np.dot([a[t, f] for f in range(self.n_features)], x) >= b[t] + self.epsilon))
                            t = 2*t + 1
                        else:
                            temp_2 = "gamma_" + str(gamma_con) + " == 1 >> "
                            for f in range(self.n_features):
                                temp_2 = temp_2 + str("a[" + str(t) + "," + str(f) + "]*" + str(x[f]) + " + ")

                            temp_2 = temp_2[:-3] + " >= " + "b[" + str(t) + "] +  epsilon"
                            if self.verbose:
                                print(temp_2)
                            
                            model.addConstr((gamma[1] == 1) >> (np.dot([a[t, f] for f in range(self.n_features)], x) >= b[t] + self.epsilon))
                            model.addConstr((gamma[1] == 0) >> (np.dot([a[t, f] for f in range(self.n_features)], x) <= b[t] - self.epsilon))
                            t = 2*t
                        
                        benders_cut_rhs.add(gamma[1])
                    benders_cut_rhs.add(c[t, self.y[x_perturb[0]]])
                    temp = temp + "c[" + str(t) + "," + str(self.y[x_perturb[0]]) + "] + "

                if self.verbose:
                    print(temp[:-3])
                
                model.addConstr(g.sum() <= benders_cut_rhs)
                model.optimize()
            else:
                break

            a_vals = model.getAttr('X', a)
            b_vals = model.getAttr('X', b)
            c_vals = model.getAttr('X', c)

            acc_count = 0
            for i, x in enumerate(self.X):
                t = 1
                while t <= len(self.branch_nodes):
                    if (np.dot([a_vals[t, f] for f in range(self.n_features)], x) > b_vals[t] + self.epsilon) or (np.abs(np.dot([a_vals[t, f] for f in range(self.n_features)], x) - b_vals[t] - self.epsilon) <= self.epsilon/100):
                        t = 2*t + 1
                    else:
                        t = 2*t
                        
                if c_vals[t, self.y[i]] > 0.5:
                    acc_count += 1

            if self.verbose:
                print()
                print("Hyperplanes:")
                for n in self.branch_nodes:
                    temp = str(n) + ": "
                    for i in range(self.n_features):
                        temp = temp + str(a_vals[n, i]) + "x" + str(i) + " + "
                    temp = temp[:-3] + " <= " + str(b_vals[n])
                    print(temp)

                print()
                print("Leaf nodes labels:")
                for n in self.leaf_nodes:
                    temp = str(n) + ": "
                    for i in self.classes:
                        if c_vals[n, i] == 1:
                            temp = temp + str(i)
                    print(temp)

                print()
            
            time_stats_row.append(acc_count/len(self.X))
            time_stats_row.append(model.objVal)
            time_stats_row.append(model.Runtime)

            self.time_stats.at[iter_count] = time_stats_row

            print("Accuracy: ", acc_count/len(self.X))
            print("Objective Value: ", model.objVal)
            print("---------------------------------------------------------")
            print()