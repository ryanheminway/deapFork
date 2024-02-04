# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:01:35 2023

@author: hemin
"""

from eant import *
from eant_visualization import *

from deap import creator, base, tools
import eant_algorithm
import pandas as pd
import numpy as np
import random
import os 
import torch

from pathlib import Path
import time
import pickle

import math

# --------------- (START) Random tests of basic functionality ------------- #
def rand_tests():   
    g2 = []
    g2.append((Neuron(0, 2, depth=0), 1))
    g2.append((Jumper(3), 1))
    g2.append((Neuron(1, 1, depth=1), 1))
    g2.append((Jumper(3), 1))
    g2.append((Neuron(2, 1, depth=0), 1))
    g2.append((Neuron(3, 2, depth=1), 1))
    g2.append((InputNode("x"), 1))
    g2.append((InputNode("y"), 1))
    g2 = Genome(g2)

    #export_eant_tree(g2, file="TestEANTVisualization.png")

    genomes = []
    for i in range(5):
        genomes.append(generate(2, "x", "y"))
        
    for i in range(10):
        print ("mutate iter: ", i)
        for j in range(len(genomes)):
            ind = genomes[j]
            genomes[j] = mutateCGE(ind, ind_pb=0.5)
            
        
    s = 0
    i = 0
    for ind in genomes:
        export_eant_tree(ind, file="ind_" + str(i) + ".png")
        i +=1 
        for ind_2 in genomes:
            if (len(ind) <= 6 or len(ind_2) <= 6):
                continue
            if (ind == ind_2):
                continue
            if (are_graphs_similar(ind, ind_2)):
                s += 1
                print ("Got a match: ", s)
                
                export_eant_tree(ind, file="match_" + str(s) + "_ind_1.png")
                export_eant_tree(ind_2, file="match_" + str(s) + "_ind_2.png")

# --------------- (END) Random tests of basic functionality ------------- #


# ------------------------ (START) Solving XOR  ------------------------- #

def evaluate_xor(individual):
    """
    Fitness function for XOR problem. Tests model on the 4 standard cases for
    XOR.

    """
    def classify(result):
        """
        Classify the neuron activation value from an EANT individual. This
        classification method expects all activation values in the range
        0-1.

        """
        if (result[0] > 0.5):
            return 1.0
        else:
            return 0.0
    
    
    func = individual.evaluate
    fitness = 0
    if(classify(func(a=0.0, b=0.0)) == 0.0):
        fitness += 1
    if(classify(func(a=1.0, b=0.0)) == 1.0):
        fitness += 1
    if(classify(func(a=0.0, b=1.0)) == 1.0):
        fitness += 1
    if(classify(func(a=1.0, b=1.0)) == 0.0):
        fitness += 1
        
    return fitness,

def evaluate_xor_small_network(individual):
    """
    Alternative fitness function attempting to incentivise the algorithm to
    find networks which use fewer connections. 

    """
    def classify(result):
        if (result[0] > 0.5):
            return 1.0
        else:
            return 0.0
    #print("Evaluating: ", individual)
    #export_eant_tree(individual, file="lastIndividual.png")
    func = individual.evaluate
    fitness = 0
    if(classify(func(a=0.0, b=0.0)) == 0.0):
        fitness += 1
    if(classify(func(a=1.0, b=0.0)) == 1.0):
        fitness += 1
    if(classify(func(a=0.0, b=1.0)) == 1.0):
        fitness += 1
    if(classify(func(a=1.0, b=1.0)) == 0.0):
        fitness += 1
    
    fitness *= 400 # arbitrary
    fitness += 3 # Fitness cap will be 16 (all networks start with size 3)
    fitness -= len(individual)
    return fitness,

def eant_solve_xor():
    """
    This function demonstrates solving the XOR classical problem using the
    EANT algorithm. 
    """
    outputs = 1
    inputs = ['a', 'b']
    n_gens = 40
    n_pop = 25
    mut_rate = 0.1
    
    creator.create("FitnessMax", base.Fitness, weights=(1,))
    creator.create("Individual", Genome, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("generate", generate, outputs, *inputs)
    toolbox.register("individual", creator.Individual, toolbox.generate())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_xor)
    toolbox.register("mutate", mutate_cge)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    #print("individual: ", toolbox.individual)
    
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)
    
    pop, _ = eant_algorithm.eant_algorithm(pop, toolbox, starting_mutpb=mut_rate, ngen=n_gens,
                                           stats=stats, halloffame=hof, verbose=True)
    
    best = hof[0]
    export_eant_tree(best, file="xor_solver_eant.png")
    print("Got XOR Evaluation on best: ", evaluate_xor(best))

# ------------------------ (END) Solving XOR  ------------------------- #


# ----------------- (START) Solving Iris Classification  ------------------ #

def class_to_vec(sample_class):
    """
    Helper method to create a one-hot vector (PyTorch tensor) when given a 
    string matching a class in the Iris dataset.
    """
    type_dict = { 'iris-setosa' : [1, 0, 0],
                  'iris-virginica' : [0, 1, 0], 
                  'iris-versicolor' : [0, 0, 1],
                  }
    s_c = sample_class.lower()
    return torch.tensor(type_dict[s_c])

def class_to_int(sample_class):
    """
    Helper method to create a integer based on a given class string.
    """
    type_dict = { 'iris-setosa' : 0,
                  'iris-virginica' : 1, 
                  'iris-versicolor' : 2,
                  }
    s_c = sample_class.lower()
    return torch.tensor(type_dict[s_c])

# Iris dataset from UCI MLR (added header manually)
# s_len = Sepal Length (cm)
# s_wid = Sepal Width (cm)
# p_len = Petal Length (cm)
# p_wid = Petal Width (cm)
# c = classification (Iris-setosa', 'Iris-virginica', 'Iris-versicolour')
iris_dataset = pd.read_csv("./../../irisDataset/irisDataset.csv")
print(iris_dataset.describe(include='all')) # include all for string class

msk = np.random.rand(len(iris_dataset)) < 0.8
train = iris_dataset[msk]
holdout = iris_dataset[~msk]
# check the number of records we'll validate our MSE with
print(holdout.describe())
# check the number of records we'll train our algorithm with
print(train.describe())

S_LEN = train.s_len.values # torch.from_numpy(train.s_len.values).float()
S_WID  = train.s_wid.values # torch.from_numpy(train.s_wid.values).float()
P_LEN = train.p_len.values # torch.from_numpy(train.p_len.values).float()
P_WID = train.p_wid.values # torch.from_numpy(train.p_wid.values).float()
Y = train.c.values  # this is our target, now mapped to Y

# Testing normalizing
S_LEN = (S_LEN - S_LEN.mean()) / S_LEN.std()
S_WID = (S_WID - S_WID.mean()) / S_WID.std()
P_LEN = (P_LEN - P_LEN.mean()) / P_LEN.std()
P_WID = (P_WID - P_WID.mean()) / P_WID.std()

def test_accuracy(individual, dataset):
    """
    Given an individual, evaluate its perfomance on a dataset and return
    the accuracy. 
    """
    func = individual.evaluate
    
    S_LEN_h = dataset.s_len.values # torch.from_numpy(train.s_len.values).float()
    S_WID_h  = dataset.s_wid.values # torch.from_numpy(train.s_wid.values).float()
    P_LEN_h = dataset.p_len.values # torch.from_numpy(train.p_len.values).float()
    P_WID_h = dataset.p_wid.values # torch.from_numpy(train.p_wid.values).float()
    # Testing normalizing
    S_LEN_h = (S_LEN_h - S_LEN_h.mean()) / S_LEN_h.std()
    S_WID_h = (S_WID_h - S_WID_h.mean()) / S_WID_h.std()
    P_LEN_h = (P_LEN_h - P_LEN_h.mean()) / P_LEN_h.std()
    P_WID_h = (P_WID_h - P_WID_h.mean()) / P_WID_h.std()
    
    Yp_list_tensors = []
    # Get predicted labels
    for i in range(len(S_LEN_h)):
        _sl = S_LEN_h[i]
        _sw = S_WID_h[i]
        _pl = P_LEN_h[i]
        _pw = P_WID_h[i]
        
        result = func(sl=_sl, sw=_sw, pl=_pl, pw=_pw)
        result = torch.from_numpy(np.array(result)).float()
        # (NOTE Ryan) Softmax done during evaluation for EANT
        softmax = torch.nn.Softmax(dim=0)
        result = softmax(result)
        Yp_list_tensors.append(result)
    
    Yp_list = [np.argmax(x.numpy()) for x in Yp_list_tensors]
    # True labels
    labels_tensors = list(map(class_to_int, dataset.c.values))
    labels = [x.item() for x in labels_tensors]
    matches = 0
    for i in range(len(labels)):
        if labels[i] == Yp_list[i]:
            matches += 1
    accuracy = matches / len(labels)
    return accuracy

def evaluate_iris(ind):
    """
    Fitness function for creating an Iris classifier using EANT. Tests an
    individual on the entire training set and returns the cross entropy loss.
    """
    global S_LEN, S_WID, P_LEN, P_WID, Y
    
    func = ind.evaluate
    result = func(sl=S_LEN[0], sw=S_WID[0], pl=P_LEN[0], pw=P_WID[0])
    # print("Result: ", result)
    # print("Torch result: ", torch.from_numpy(np.array(result)).float())
    Yp_list = []
    # Get predicted labels
    for i in range(len(S_LEN)):
        _sl = S_LEN[i]
        _sw = S_WID[i]
        _pl = P_LEN[i]
        _pw = P_WID[i]
        
        result_1 = func(sl=_sl, sw=_sw, pl=_pl, pw=_pw)

        result_2 = torch.from_numpy(np.array(result_1)).float()

        # (NOTE Ryan) Softmax done during evaluation for EANT
        softmax = torch.nn.Softmax(dim=0)
        result = softmax(result_2)
        if torch.isnan(result).any():
            print("Got result: ", result_1)
            print("Got result numpy torch: ", str(result_2))
            print("How did we get nan?")
        Yp_list.append(result)
    
    Yp = torch.Tensor(len(Yp_list), 3).float()
    #print("Yp_list: ", Yp_list)
    torch.stack(Yp_list, dim=0, out=Yp) # Turn list of tensors into stacked tensor
    #print("Yp: ", Yp)
    # True labels
    labels = list(map(class_to_int, Y))
    #print("labels: ", labels)
    labels_t = torch.Tensor(len(labels), 3).long()
    torch.stack(labels, dim=0, out=labels_t)
    #print("Yp: ", labels_t)
    # Cross entropy loss as fitness
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    #print("loss: ", loss(Yp, labels_t))
    if math.isnan(loss(Yp, labels_t).item()):
        print("How did we get nan?")
    return loss(Yp, labels_t).item(),
    
def eant_solve_iris():
    classifier = True
    guided = True
    outputs = 3
    inputs = ['sl', 'sw', 'pl', 'pw']
    n_gens = 1000 # 1000
    n_pop = 50
    mut_rate = 0.05
    iters = 100
    
    today = time.strftime("%Y%m%d")
    run_dir = "runs"
    model_path = str(Path.cwd()) + "/" + run_dir + "/" + today + '_iris' 
    if guided:
        model_path += "_guided"
    model_path += "/"
    Path(model_path).mkdir(parents=True, exist_ok=True)

    results_file = model_path + '/results.txt'
    def _write_to_file(file, content):
        f = open(file, 'a')
        f.write(content)  
        f.close()
        
    _write_to_file(results_file, "Running EANT solver for Iris Classification\n")
    
    creator.create("FitnessMin", base.Fitness, weights=(-1,))
    creator.create("Individual", Genome, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("generate", generate, outputs, guided, classifier, *inputs)
    toolbox.register("individual", creator.Individual, toolbox.generate())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_iris)
    toolbox.register("mutate", mutate_cge)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("getElites", tools.selBest)
    
    for i in range(iters):
        print("Running iteration: ", i)
        pop = toolbox.population(n=n_pop)
        hof = tools.HallOfFame(3)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("med", np.median)
        
        pop, log = eant_algorithm.eant_algorithm(pop, toolbox, starting_mutpb=mut_rate, ngen=n_gens,
                                               stats=stats, halloffame=hof, verbose=True)
        
        best = hof[0]
        train_acc = test_accuracy(best, train)
        test_acc = test_accuracy(best, holdout)
        print("Got Iris Train Accuracy on best: ", train_acc)
        print("Got Iris Test Accuracy on best: ", test_acc)
        _write_to_file(results_file, "Iter {} got train / test accuracies: {} / {} \n".format(i, train_acc, test_acc))
    
        # Add final train / test accuracies as a non-series row
        record = { "train_acc" : train_acc, "test_acc" : test_acc }
        log.record(gen=n_gens + 1, nevals=0, **record)
        # Save statistics as a pickle object file on disk
        pkl_file = open(model_path + "stats_iter_{}.pickle".format(i), 'wb')
        pickle.dump(log, pkl_file)
    _write_to_file(results_file, "Finished EANT solver iterations!")    


# ----------------- (END) Solving Iris Classification  ------------------ #


# ----------------- (START) Solving Glass Classification  ------------------ #
# Glass dataset from UCI MLR (added header manually)
# RI = Refractive Index
# Na = Sodium content
# Mg = Magnesium content
# Al = Aluminum content
# Si = Silicon content
# K = Potassium content
# Ca = Calcium content
# Ba = Barium content
# Fe = Iron content
# class_id = classification (1-7)
# glass_dataset = pd.read_csv("./../../glassDataset/glassDataset.csv")
# print(glass_dataset.describe(include='all')) # include all for string class

# msk = np.random.rand(len(glass_dataset)) < 0.8
# train = glass_dataset[msk]
# holdout = glass_dataset[~msk]
# # check the number of records we'll validate with
# print(holdout.describe())
# # check the number of records we'll train our algorithm with
# print(train.describe())

# RI = train.RI.values # torch.from_numpy(train.RI.values).float()
# Na = train.Na.values # torch.from_numpy(train.Na.values).float()
# Mg = train.Mg.values # torch.from_numpy(train.Mg.values).float()
# Al = train.Al.values # torch.from_numpy(train.Al.values).float()
# Si = train.Si.values # torch.from_numpy(train.Si.values).float()
# K = train.K.values # torch.from_numpy(train.K.values).float()
# Ca = train.Ca.values # torch.from_numpy(train.Ca.values).float()
# Ba = train.Ba.values # torch.from_numpy(train.Ba.values).float()
# Fe = train.Fe.values # torch.from_numpy(train.Fe.values).float()
# Y = train.class_id.values  # this is our target, now mapped to Y

# # Testing normalizing
# RI = (RI - RI.mean()) / RI.std()
# Na = (Na - Na.mean()) / Na.std()
# Mg = (Mg - Mg.mean()) / Mg.std()
# Al = (Al - Al.mean()) / Al.std()
# Si = (Si - Si.mean()) / Si.std()
# K = (K - K.mean()) / K.std()
# Ca = (Ca - Ca.mean()) / Ca.std()
# Ba = (Ba - Ba.mean()) / Ba.std()
# Fe = (Fe - Fe.mean()) / Fe.std()

def test_glass_acc(individual, dataset):
    """
    Given an individual, evaluate its perfomance on a dataset and return
    the accuracy. 
    """
    func = individual.evaluate
    
    RI_h = dataset.RI.values # torch.from_numpy(train.RI.values).float()
    Na_h = dataset.Na.values # torch.from_numpy(train.Na.values).float()
    Mg_h = dataset.Mg.values # torch.from_numpy(train.Mg.values).float()
    Al_h = dataset.Al.values # torch.from_numpy(train.Al.values).float()
    Si_h = dataset.Si.values # torch.from_numpy(train.Si.values).float()
    K_h = dataset.K.values # torch.from_numpy(train.K.values).float()
    Ca_h = dataset.Ca.values # torch.from_numpy(train.Ca.values).float()
    Ba_h = dataset.Ba.values # torch.from_numpy(train.Ba.values).float()
    Fe_h = dataset.Fe.values # torch.from_numpy(train.Fe.values).float()

    # Testing normalizing
    RI_h = (RI_h - RI_h.mean()) / RI_h.std()
    Na_h = (Na_h - Na_h.mean()) / Na_h.std()
    Mg_h = (Mg_h - Mg_h.mean()) / Mg_h.std()
    Al_h = (Al_h - Al_h.mean()) / Al_h.std()
    Si_h = (Si_h - Si_h.mean()) / Si_h.std()
    K_h = (K_h - K_h.mean()) / K_h.std()
    Ca_h = (Ca_h - Ca_h.mean()) / Ca_h.std()
    Ba_h = (Ba_h - Ba_h.mean()) / Ba_h.std()
    Fe_h = (Fe_h - Fe_h.mean()) / Fe_h.std()
    
    Yp_list_tensors = []
    # Get predicted labels
    for i in range(len(RI_h)):
        _ri = RI_h[i]
        _na = Na_h[i]
        _mg = Mg_h[i]
        _al = Al_h[i]
        _si = Si_h[i]
        _k = K_h[i]
        _ca = Ca_h[i]
        _ba = Ba_h[i]
        _fe = Fe_h[i]
        
        result = func(ri=_ri, na=_na, mg=_mg, al=_al, si=_si, k=_k, ca=_ca, ba=_ba, fe=_fe)
        result = torch.from_numpy(np.array(result)).float()
        # (NOTE Ryan) Softmax done during evaluation for EANT
        softmax = torch.nn.Softmax(dim=0)
        result = softmax(result)
        Yp_list_tensors.append(result)
    
    Yp_list = [np.argmax(x.numpy()) for x in Yp_list_tensors]
    # True labels
    labels = [x-1 for x in dataset.class_id.values]
    matches = 0
    for i in range(len(labels)):
        if labels[i] == Yp_list[i]:
            matches += 1
    accuracy = matches / len(labels)
    return accuracy

def evaluate_glass(ind):
    """
    Fitness function for creating an Glass classifier using EANT. Tests an
    individual on the entire training set and returns the cross entropy loss.
    """
    global RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, Y
    
    func = ind.evaluate
    Yp_list = []
    # Get predicted labels
    for i in range(len(RI)):
        _ri = RI[i]
        _na = Na[i]
        _mg = Mg[i]
        _al = Al[i]
        _si = Si[i]
        _k = K[i]
        _ca = Ca[i]
        _ba = Ba[i]
        _fe = Fe[i]
        
        result_1 = func(ri=_ri, na=_na, mg=_mg, al=_al, si=_si, k=_k, ca=_ca, ba=_ba, fe=_fe)

        result_2 = torch.from_numpy(np.array(result_1)).float()

        # (NOTE Ryan) Softmax done during evaluation for EANT
        softmax = torch.nn.Softmax(dim=0)
        result = softmax(result_2)
        if torch.isnan(result).any():
            print("Got result: ", result_1)
            print("Got result numpy torch: ", str(result_2))
            print("How did we get nan?")
        Yp_list.append(result)
    
    Yp = torch.Tensor(len(Yp_list), 3).float()
    #print("Yp_list: ", Yp_list)
    torch.stack(Yp_list, dim=0, out=Yp) # Turn list of tensors into stacked tensor
    #print("Yp: ", Yp)
    # True labels
    labels = [torch.tensor(x-1) for x in Y]
    #print("labels: ", labels)
    labels_t = torch.Tensor(len(labels), 3).long()
    torch.stack(labels, dim=0, out=labels_t)
    #print("Yp: ", labels_t)
    # Cross entropy loss as fitness
    loss = torch.nn.CrossEntropyLoss(reduction='sum')
    #print("loss: ", loss(Yp, labels_t))
    if math.isnan(loss(Yp, labels_t).item()):
        print("How did we get nan?")
    return loss(Yp, labels_t).item(),
    
def eant_solve_glass():
    classifier = True
    guided = True
    outputs = 7
    inputs = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe']
    n_gens = 1000 # 1000
    n_pop = 50
    mut_rate = 0.05
    iters = 100
    
    today = time.strftime("%Y%m%d")
    run_dir = "runs"
    model_path = str(Path.cwd()) + "/" + run_dir + "/" + today + '_glass' 
    if guided:
        model_path += "_guided"
    model_path += "/"
    Path(model_path).mkdir(parents=True, exist_ok=True)

    results_file = model_path + '/results.txt'
    def _write_to_file(file, content):
        f = open(file, 'a')
        f.write(content)  
        f.close()
        
    _write_to_file(results_file, "Running EANT solver for Glass Classification\n")
    
    creator.create("FitnessMin", base.Fitness, weights=(-1,))
    creator.create("Individual", Genome, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("generate", generate, outputs, guided, classifier, *inputs)
    toolbox.register("individual", creator.Individual, toolbox.generate())
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_glass)
    toolbox.register("mutate", mutate_cge)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("getElites", tools.selBest)
    
    for i in range(iters):
        print("Running iteration: ", i)
        pop = toolbox.population(n=n_pop)
        hof = tools.HallOfFame(3)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("med", np.median)
        
        pop, log = eant_algorithm.eant_algorithm(pop, toolbox, starting_mutpb=mut_rate, ngen=n_gens,
                                               stats=stats, halloffame=hof, verbose=True)
        
        best = hof[0]
        train_acc = test_glass_acc(best, train)
        test_acc = test_glass_acc(best, holdout)
        print("Got Glass Train Accuracy on best: ", train_acc)
        print("Got Glass Test Accuracy on best: ", test_acc)
        _write_to_file(results_file, "Iter {} got train / test accuracies: {} / {} \n".format(i, train_acc, test_acc))
    
        # Add final train / test accuracies as a non-series row
        record = { "train_acc" : train_acc, "test_acc" : test_acc }
        log.record(gen=n_gens + 1, nevals=0, **record)
        # Save statistics as a pickle object file on disk
        pkl_file = open(model_path + "stats_iter_{}.pickle".format(i), 'wb')
        pickle.dump(log, pkl_file)
    _write_to_file(results_file, "Finished EANT solver iterations!")    


# ----------------- (END) Solving Glass Classification  ------------------ #

if __name__ == "__main__":
    # rand_tests()
    
    #eant_solve_xor()
    
    eant_solve_iris()
    
    #eant_solve_glass()
