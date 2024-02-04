# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:24:31 2023

@author: Ryan Heminway

Implementation of the EANT algorithm.
"""
import random

from deap import tools
from eant import are_graphs_similar, Genome
from deap import creator # ,cma
import collections
import copy

import cma

import numpy

def varMutateOnly(population, toolbox, mutpb):
    """
    Vary a population by applying mutation (structural mutation) to all 
    Neurons in the population, with some probability mutpb.

    Parameters
    ----------
    population : List
        List representing a current population of individual solutions.
    toolbox : deap.base.Toolbox
        Toolbox to use for applying mutation.
    mutpb : Float
        Float representing chance that any Neuron in an individual is mutated.

    Returns
    -------
    offspring : List
        A new population of Genomes, that has been mutated.

    """
    offspring = [toolbox.clone(ind) for ind in population]
    for i in range(len(offspring)):
        offspring[i] = creator.Individual(toolbox.mutate(offspring[i], mutpb))
        del offspring[i].fitness.values

    return offspring



class Cluster(object):
    """
    Represents a species in an EANT population, characterized by all 
    individuals in the cluster having similar (identical) network topologies.
    This allows them to be grouped, and have their weights optimized at once.
    One representative from a cluster is used to optimize weights. At the 
    end of any generation, all individuals of the cluster have their weights
    updated to match the representative. Each cluster has a tracked CMA Strategy
    object which gets updated as weights are optimized. As long as the Cluster 
    exists in the population, weights continue to get optimized from the same
    Strategy. 
    """
    def __init__(self, representative, strategy):
        self.rep = creator.Individual(representative.clone())
        self.strat = copy.copy(strategy)
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
    Run the CMA-ES algorithm to update the weights of a given Cluster. All
    individuals in the given cluster will have their weights mutated by this
    method.

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
        g = cluster.rep.clone() # Genome(cluster.rep)
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


def _get_fitness_and_neurons(population, toolbox):
    """
    Helper method to evaluate the fitnesses of a population and count their 
    total Neurons.
    
    Return: Tuple
        Result tuple indicating (best_fitness, neuron_count)
    """
    best_fit = float('-inf')
    neuron_count = 0
    # STEP 0: Evaluate the individuals with an invalid fitness
    for ind in population:
        # Track how long new structures have been around
        ind.step_protection()
        # Track # Neurons in population
        neuron_count += len(ind.neuron_list)
        
        if not ind.fitness.valid:
            fitness = toolbox.evaluate(ind)
            ind.fitness.values = fitness
        
        # TODO only applicable to single-objective fitness atm
        if ind.fitness.values[0] > neuron_count:
            neuron_count = ind.fitness.values[0]
            
    return neuron_count, neuron_count


def _make_clusters(population, clusters, gens_protected):
    """
    Helper method to update a list of clusters based on a given population.
    This method will create clusters out of a population. This method
    will mutate the given list `clusters`.

    """
    for c in clusters:
        c.clear_members() # Stop tracking last gen's pop
    for ind in population:
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
            # Found unique structure, create a brand new Cluster for it.
            # Setting initial distribution of weights to be gaussian around
            # 0 with STD of .3
            N = len(ind)
            if ind.parent_cluster == None or True:
                new_strat = cma.Strategy(centroid=[0.0]*N, sigma=0.3)
            else:
                # TODO This CMA adapation of parent cluster is not working well
                
                print("C matrix: ", ind.parent_cluster.strat.C)
                print("ind len: ", len(ind))
                print("C len: ", len(ind.parent_cluster.strat.C))
                # Use old entries in new CMA
                # TODO Not sure this is the way they did it in the original EANT paper
                size_diff = N - len(ind.parent_cluster.strat.C)
                old_size = len(ind.parent_cluster.strat.C)
                new_cmatrix = ind.parent_cluster.strat.C
                new_centroid = ind.parent_cluster.strat.centroid
                new_sigma = ind.parent_cluster.strat.sigma
                if (size_diff >= 0):
                    new_c = numpy.identity(N)
                    centr = [0.0] * N 
                    # Calculate which parts of old C matrix can be used, based on
                    # what was gained or lost in mutation
                    last_added = -1
                    added = 0
                    for added_idx in ind.indices_added:
                        print("Last_added: ", last_added)
                        print("Added: ", added_idx)
                        print("Shape: ", new_cmatrix[:, (last_added + 1 - added):added_idx].shape)
                        new_c[:old_size, last_added + 1:added_idx] = new_cmatrix[:, (last_added + 1):added_idx]
                        centr[last_added+1:added_idx] = new_centroid[(last_added + 1):added_idx]
                        last_added = added_idx
                        added += 1
                    new_c[:old_size, last_added + 1:] = new_cmatrix[:, (last_added + 1 - added):]
                    centr[last_added+1:] = new_centroid[(last_added + 1 - added):]
                    new_centroid= centr
                    new_cmatrix = new_c
                    
                elif (size_diff < 0):
                    new_c = numpy.identity(N)
                    centr = [0.0] * N 
                    last_lost = -1
                    added = 0
                    for lost_idx in ind.indices_lost:
                        new_c[:, last_lost + 1:lost_idx] = new_cmatrix[:N, last_lost + 1:lost_idx]
                        centr[last_lost+1:lost_idx] = new_centroid[last_lost+1:lost_idx]
                        last_lost = lost_idx
                        added += 1
                    new_c[:, last_lost + 1:] = new_cmatrix[:N, last_lost + 1 - added:]
                    centr[last_lost+1:] = new_centroid[last_lost+1-added:]
                    new_centroid = centr
                    new_cmatrix = new_c
                    
                    # new_cmatrix = new_cmatrix[:N, :N]
                    
                print("New C: ", new_cmatrix)
                test = numpy.identity(N)
                print("NEW C SHAPE: ", new_cmatrix.shape)
                print("ID SHAPE: ", test.shape)
                # Use parent cluster as starting Covariance Matrix
                #new_strat = cma.Strategy(centroid=[0.0]*N, sigma=0.3, cmatrix=new_cmatrix)
                new_strat = cma.Strategy(centroid=new_centroid, sigma=new_sigma, cmatrix=new_cmatrix)
            new_cluster = Cluster(representative=ind, strategy=new_strat)
            if not new_cluster.add_member(ind):
                print("This should never happen")
                exit(1)
            # Mark individual as protected
            ind.set_protection(gens_protected)
            clusters.append(new_cluster)


def _do_mutation(population, mut_rate, toolbox):
    # "Protected" individuals represent new toplogies not yet optimized
    protected = [ind for ind in population if ind.is_protected()]
    unprotected = [ind for ind in population if not ind.is_protected()]
    print("mutate prob: ", mut_rate)
    # Structural mutation
    new_offspring = []
    new_individuals = varMutateOnly(unprotected, toolbox, mut_rate)
    for i in new_individuals:
        new_offspring.append(i)
    for i in protected:
        # TODO Would be better to avoid mutating individual from an outside process like this 
        # Need to reset previous mutation tracking for individuals that don't get mutated this generation
        i.indices_added = []
        i.indices_lost = []
        new_offspring.append(i)
    return new_offspring


# TODO Last few init params are arbitrarily set... no params given in EANT paper
def eant_algorithm(population, toolbox, starting_mutpb, ngen, stats=None,
                   halloffame=None, verbose=False, fit_prog_gens=6, 
                   fit_growth_threshold=0.2, gens_protected=10, n_elites=3):
    """
    Run the EANT evolutionary algorithm as described by the paper: 
        https://web.archive.org/web/20070613093500/http://www.ks.informatik.uni-kiel.de/~yk/ESANN2005EANT.pdf

    Parameters
    ----------
    population : List
        List of Creator.Individuals (Genomes with a Fitness attribute). Represents
        starting population for the algorithm.
    toolbox : deap.base.Toolbox
        Toolbox of evolutionary computational helper methods.
    starting_mutpb : Float
        Probability to apply mutation to any Neuron in the population-to-be-mutated.
        Represents the starting probability. Probability will change over time
        as Neurons get added to the population.
    ngen : Integer
        Number of Generations to run for
        
    fit_prog_gens : Integer, optional
        The number of generations to span when looking at fitness progression.
        The default is 6. Structural mutation is initiated whenever fitness
        progression has stagnated.
    fit_growth_threshold : Float, optional
        The threshold for how much fitness must grow in order to represent real
        progress. The default is 0.2.
    gens_protected : Integer, optional
        Number of generations to protect a new structure in the population. This
        gives a new topology time to optimize its weights before competing with
        the entire population. The default is 10.
    n_elites : Integer, optional
        Number of Elites. An Elite is an individual with high fitness which will
        get transferred to the next generation's population no matter what.
        The default is 3.

    Returns
    -------
    population : List
        State of population at the end of the specified number of generations.

    """
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    # Need to know neuron count for mutation rate updates specified by EANT paper
    start_neuron_count = sum([len(ind.neuron_list) for ind in population])
    # Number of generations to span for stagnant fitness checks
    last_n_fitnesses = collections.deque([0 for i in range(fit_prog_gens)], maxlen=fit_prog_gens)
    
    # STEP 0: Evaluate the individuals with an invalid fitness
    best_fit_this_gen, curr_neuron_count = _get_fitness_and_neurons(population, toolbox)
    clusters = [] # Cluster objects
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # STEP 1: Select the next generation individuals
        # Find "protected" individuals which are newly added structures
        protected = [ind for ind in population if ind.is_protected()]
        elites = toolbox.getElites(population, n_elites)
        elites = [e for e in elites if e not in protected]
        offspring = toolbox.select(population, len(population) - len(protected) - len(elites))
        print("Gen: ", gen, " got proteced: ", len(protected))
        print("Gen: ", gen, " got offspring: ", len(offspring))
        offspring = offspring + protected
        print("Gen: ", gen, " got ALL offspring: ", len(offspring))
        print("Gen: ", gen, " got clusters: ", clusters)
        
        # STEP 2: Check fitness progress to determine if structural mutation happens
        new_pop = []
        last_n_fitnesses.append(best_fit_this_gen)
        if gen >= fit_prog_gens and (last_n_fitnesses[-1] - last_n_fitnesses[0] < fit_growth_threshold):
            # STEP 2a: Structural mutation
            # Calc new mut_pb based on number of neurons in population
            # Equation 3 in EANT paper
            new_mutpb = (start_neuron_count / curr_neuron_count) * starting_mutpb
            # Mutate population                        
            offspring = _do_mutation(offspring, new_mutpb, toolbox)
        
        # Elites did not get mutated
        offspring = offspring + elites
        assert(len(offspring) == len(population))
            
        # STEP 3: Evaluate the individuals with an invalid fitness
        best_fit_this_gen, curr_neuron_count = _get_fitness_and_neurons(offspring, toolbox)
            
        # STEP 4: Weights "mutation" through CMA-ES
        # Cluster individuals by structure
        _make_clusters(offspring, clusters, gens_protected)
        # Calculate total average-fitness across clusters
        tot_avg_fitness = sum([c.avg_fitness() for c in clusters])
        # Carry out CMA-ES once all are clustered
        for c in clusters:
            if c.is_empty():
                continue
            # Number of evals weighted by a cluster's avg fitness
            # Equation from EANT paper
            n_evals = int(c.avg_fitness() / tot_avg_fitness) * len(population)
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
        # Add, as an object, the best individual so far
        logbook.record(best_ind=halloffame[0])
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)
        
        # Clean out dead clusters
        clusters = [c for c in clusters if not c.is_empty()]

    return population, logbook