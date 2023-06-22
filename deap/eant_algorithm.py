# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:24:31 2023

@author: Ryan Heminway

Implementation of the EANT algorithm.
"""
import random

from deap import tools
from eant import are_graphs_similar, Genome
from deap import cma, creator
import collections
import copy

def varMutateOnly(population, toolbox, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    for i in range(len(offspring)):
        offspring[i] = creator.Individual(toolbox.mutate(offspring[i], mutpb))
        del offspring[i].fitness.values

    return offspring



class Cluster(object):
    def __init__(self, representative, strategy, protected_gens=10):
        # TODO protected_gens is arbitrary
        self.rep = creator.Individual(representative)
        self.strat = copy.copy(strategy)
        self.protected_gens = protected_gens
        self.members = []
        
    def is_empty(self):
        return len(self.members) == 0
    
    def clear_members(self):
        self.members = []
    
    def add_member(self, individual):
        if are_graphs_similar(individual, self.rep):
            self.members.append(individual)
            return True
        else:
            return False
    
    def update_weights(self, weights):
        for ind in self.members:
            ind.set_weights(weights)
            del ind.fitness.values
        self.rep.set_weights(weights)
        del self.rep.fitness.values
        
    def update_fitness(self, eval_fn):
        fitness = eval_fn(self.rep)
        self.rep.fitness.value = fitness
        for ind in self.members:
            ind.fitness.values = fitness
        
    def avg_fitness(self):
        if self.is_empty():
            return 0
        tot_fit = 0
        for member in self.members:
            # TODO single-objective fitness only
            tot_fit += member.fitness.values[0]
            
        return (tot_fit / len(self.members))


def _run_cma_es(cluster, evals, eval_fn):
    """
    Run the CMA-ES algorithm to update the weights of a given Cluster.

    Parameters
    ----------
    cluster : Cluster
        Cluster of Genome individuals to update the weights of.
    evals : Int
        Number of evaluations or updates to make to the CMA 
    eval_fn : Function object
        Function that takes an individual as input and returns the fitness
        of that individual.

    Returns
    -------
    None
    """
    if evals <= 0:
        return
    cma_strat = cluster.strat
    
    class cma_individual(list):
        def __init__(self, weights, fitness):
            list.__init__(self, weights)
            self.fitness = fitness
    
    def init_new_genome(weights):
        # Individual in CMA-ES population will be identical to representative
        # of cluster, just with unique weights
        g = Genome(cluster.rep)
        g.set_weights(weights)
        g = creator.Individual(g)
        return g
    # TODO Is this the correct interpretation of "number of evaluations"?
    for i in range(evals):
        # Generate population of candidate solutions based on CMA
        cma_pop = cma_strat.generate(init_new_genome)
        #print("cma_pop: ", cma_pop)
        
        best_fit = float('-inf')
        best_weights = []
        # Evaluate all new candidate solutions (weights in this case)
        fitnesses = map(eval_fn, cma_pop)
        for ind, fit in zip(cma_pop, fitnesses):
            ind.fitness.values = fit
            # Track best solution
            if fit[0] > best_fit:
                best_fit = fit[0]
                best_weights = ind.get_weights()
        # Update CMA according to performance
        cma_strat.update([cma_individual(i.get_weights(), i.fitness) for i in cma_pop])
    # When done, update all weights of individuals in the cluster to be
    # that of best found individual from CMA-ES    
    cluster.update_weights(best_weights)
    # For tracking, update all fitnesses in the cluster (they will be the same)
    cluster.update_fitness(eval_fn)


def eant_algorithm(population, toolbox, starting_mutpb, ngen, stats=None,
                   halloffame=None, verbose=False):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    
    # Need to know neuron count for mutation rate updates specified by
    # EANT paper
    start_neuron_count = 0
    for ind in population:
        start_neuron_count += len(ind.neuron_list)
    
    # TODO n is totally arbitrary atm, EANT paper does not indicate which value is used
    # Number of generations to span for stagnant fitness checks
    n = 6
    last_n_fitnesses = collections.deque([0 for i in range(n)], maxlen=n)
    # TODO This threshold is also arbitrary... not sure how to set
    fit_growth_threshold = 0.25
    # Generations to protect new structures
    # TODO This is also arbitrary...
    gens_protected = 4
    
    
    best_fit_this_gen = float('-inf')
    curr_neuron_count = 0
    # STEP 0: Evaluate the individuals with an invalid fitness
    for ind in population:
        # Track how long new structures have been around
        ind.step_protection()
        # Track # Neurons in population
        curr_neuron_count += len(ind.neuron_list)
        
        if not ind.fitness.valid:
            fitness = toolbox.evaluate(ind)
            ind.fitness.values = fitness
        
        # TODO only applicable to single-objective fitness atm
        if ind.fitness.values[0] > best_fit_this_gen:
            best_fit_this_gen = ind.fitness.values[0]
    
    
    clusters = [] # Cluster objects
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # STEP 1: Select the next generation individuals
        # Find "protected" individuals which are newly added structures
        protected = [ind for ind in population if ind.is_protected()]
        offspring = toolbox.select(population, len(population) - len(protected))
        print("Gen: ", gen, " got proteced: ", len(protected))
        print("Gen: ", gen, " got offspring: ", len(offspring))
        
        offspring = offspring + protected
        print("Gen: ", gen, " got ALL offspring: ", len(offspring))
        print("Gen: ", gen, " got clusters: ", clusters)
        
        # STEP 2: Check fitness progress to determine type of mutation in this gen
        new_pop = []
        last_n_fitnesses.append(best_fit_this_gen)
        if gen >= n and (last_n_fitnesses[-1] - last_n_fitnesses[0] < fit_growth_threshold):
            # STEP 2a: Structural mutation
            # Calc new mut_pb based on number of neurons in population
            # Equation 3 in EANT paper
            new_mutpb = (start_neuron_count / curr_neuron_count) * starting_mutpb
            # "Protected" individuals represent new toplogies not yet optimized
            protected = [ind for ind in offspring if ind.is_protected()]
            unprotected = [ind for ind in offspring if not ind.is_protected()]
            print("mutate prob: ", new_mutpb)
            # Structural mutation
            new_offspring = []
            new_individuals = varMutateOnly(unprotected, toolbox, new_mutpb)
            for i in new_individuals:
                new_offspring.append(i)
            for i in protected:
                new_offspring.append(i)
            offspring = new_offspring
            
            
        # STEP 3: Evaluate the individuals with an invalid fitness
        curr_neuron_count = 0
        best_fit_this_gen = float('-inf')
        for ind in offspring:
            # Track how long new structures have been around
            ind.step_protection()
            # Track # Neurons in population
            curr_neuron_count += len(ind.neuron_list)
            
            if not ind.fitness.valid:
                fitness = toolbox.evaluate(ind)
                ind.fitness.values = fitness
            
            # TODO only applicable to single-objective fitness atm
            if ind.fitness.values[0] > best_fit_this_gen:
                best_fit_this_gen = ind.fitness.values[0]
            
            
        # STEP 4: Weights "mutation" through CMA-ES
        # Cluster individuals by structure
        for c in clusters:
            c.clear_members() # Stop tracking last gen's pop
        for ind in offspring:
            clustered = False
            # Look for an existing Cluster for this individual
            for c in clusters:
                # Add individual if its an isomorphic match
                if c.add_member(ind):
                    # Track as parent
                    ind.parent_cluster = c
                    clustered = True
                    break
            if not clustered:
                #print("Got a unique structure")
                # Found unique structure, create a brand new Cluster for it.
                # Setting initial distribution of weights to be gaussian around
                # 0 with STD of 1
                N = len(ind)
                if ind.parent_cluster == None:
                    new_strat = cma.Strategy(centroid=[0.0]*N, sigma=1.0)
                else:
                    # Use parent cluster as starting Covariance Matrix
                    new_strat = cma.Strategy(centroid=[0.0]*N, sigma=1.0, cmatrix=ind.parent_cluster.C)
                new_cluster = Cluster(representative=ind, strategy=new_strat)
                new_cluster.add_member(ind)
                # Mark individual as protected
                ind.set_protection(gens_protected)
                clusters.append(new_cluster)
        # Calculate total average-fitness across clusters
        tot_avg_fitness = sum([c.avg_fitness() for c in clusters])
        # Carry out CMA-ES once all are clustered
        for c in clusters:
            if c.is_empty():
                continue
            # Number of evals weighted by a cluster's avg fitness
            # Equation from EANT paper
            n_evals = int((c.avg_fitness() / tot_avg_fitness) * len(population))
            # Run CMA-ES and update weights of all individuals in this cluster
            _run_cma_es(c, n_evals, toolbox.evaluate)
            for member in c.members:
                new_pop.append(member)
                
        # Replace the current population by the offspring
        population[:] = new_pop
        
        # Update the hall of fame
        if halloffame is not None:
            halloffame.update(population)
            
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)
            
        # Short circuit if we found a solution
        if toolbox.evaluate(halloffame[0])[0] == 4:
            return population, logbook
        
        # Clean out dead clusters
        clusters = [c for c in clusters if not c.is_empty()]

    return population, logbook