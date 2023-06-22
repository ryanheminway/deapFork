# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:01:35 2023

@author: hemin
"""

from eant import *
from eant_visualization import *

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

def evaluate_xor(individual):
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
        
    return fitness,

def evaluate_xor_small_network(individual):
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


if __name__ == "__main__":
    # rand_tests()
    
    # genome = []
    # genome.append((Neuron(0, 3), 1))
    # genome.append((InputNode("x"), 1))
    # genome.append((InputNode("y"), 1))
    # genome.append((Jumper(0, recurrent=True), 1))
    
    # recursive_g = Genome(genome)
    # print("recursive result:", recursive_g.evaluate(x=1, y=1))
    # export_eant_tree(recursive_g, file="recursive.png")
    # exit(1)
    
    # --------------------- EANT Solve XOR --------------------------- #
    from deap import creator, base, tools
    import eant_algorithm
    import numpy
    
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
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)
    
    pop, _ = eant_algorithm.eant_algorithm(pop, toolbox, starting_mutpb=mut_rate, ngen=n_gens,
                                           stats=stats, halloffame=hof, verbose=True)
    
    best = hof[0]
    export_eant_tree(best, file="xor_solver_eant.png")
    print("Got XOR Evaluation on best: ", evaluate_xor(best))

