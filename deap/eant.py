# -*- coding: utf-8 -*-
"""
@author: Ryan Heminway

This :mod:`eant` module provides the methods and classes
to perform EANT with DEAP. It essentially contains the classes
to build an individual using the Common Genetic Encoding (CGE),
and the functions to evaluate it as a neural network.
"""


from collections import deque
import numpy as np
import random


# -------------------- EANT Data Structure ------------------ #

class Node(object):
    """
    A basic node in a EANT individual (Genome). Can be either a Neuron,
    InputNode, or Jumper node. Tracks its previous output for use in recurrent
    networks.
    """
    def __init__(self, name, prev_output=0):
        self.name = name
        # Output from last timestep
        self.prev_output = 0
        # Output from current timestep
        self.curr_output = 0
        
    def tracking_value(self):
        return 1
    
    def step(self):
        """ 
        Progress a timestep by updating previous output.
        """
        self.prev_output = self.curr_output
        
    def reset(self):
        """
        Reset the state of this node. If it is to be used for a fresh forward
        pass.
        """
        self.prev_output = 0
        self.curr_output = 0
        
    def __str__(self):
        str_rep = "NODE: {} P_OUT: {} C_OUT: {}"
        return str_rep.format(self.name, self.prev_output, self.curr_output) 
    
    def is_similar(self, node):
        """
        Check if this node is similar to another given node. Useful as a helper
        function for checking graph isomorphism across two Genomes.
        """
        return NotImplemented

    
class Neuron(Node):
    """
    A Neuron is one of the node types in an EANT individual (Genome). Represents
    a classical neural network unit with an activation function and a specified
    arity of inputs. 
    
    See Genome.evaluate() for usage details.
    """
    def __init__(self, unique_id, arity, fn='relu', depth=0):
        super().__init__("Neuron{}".format(unique_id))
        self.fn = fn
        self.unique_id = unique_id
        self.arity = arity
        self.depth = depth
        
    def activate(self, val):
        if self.fn == 'relu':
            if val > 0:
                self.curr_output = val
            else:
                self.curr_output = 0
            return self.curr_output
        elif self.fn == 'sigmoid':
            self.curr_output = 1/(1 + np.exp(-val))
            # Check for overflow
            if (self.curr_output > 1):
                self.curr_output = 1
            elif (self.curr_output < 0):
                self.curr_output = 0
            return self.curr_output
        elif self.fn == 'identity':
            self.curr_output = val
            return self.curr_output
        else:
            return NotImplemented
        
    def tracking_value(self): 
        return 1 - self.arity
    
    def __str__(self):
        str_rep = super().__str__() + " ID: {} ARITY: {} ACT_FN: {} DEPTH: {}"
        return str_rep.format(self.unique_id, self.arity, self.fn, self.depth)
    
    def is_similar(self, node):
        if (not (isinstance(node, Neuron))):
            return False
        similar = (self.depth == node.depth)
        similar = similar and (self.arity == node.arity)
        similar = similar and (self.fn == node.fn)
        return similar
            

class InputNode(Node):
    """
    An InputNode is one of the node types in an EANT individual (Genome). 
    Represents an input to the neural network. 
    
    See Genome.evaluate() for usage details.
    """
    def __init__(self, name, value=None):
        super().__init__(name)
        self.value = value
        
    def __str__(self):
        str_rep = super().__str__() + " VAL: {}"
        return str_rep.format(self.value)    
    
    def set_value(self, value):
        self.value = value
        
    def is_similar(self, node):
        if (not (isinstance(node, InputNode))):
            return False
        similar = (self.name == node.name)
        similar = similar and (self.value == node.value)
        return similar

class Jumper(Node):
    """
    A Jumper node is one of the node types in an EANT individual (Genome). 
    Represents forward or recurrent connections that break the standard 
    connectivity pattern for fully-connected MLPs. 
    
    See Genome.evaluate() for usage details.
    """
    def __init__(self, unique_id, recurrent=False):
        super().__init__("Jumper{}{}".format("Rec" if recurrent else "For", unique_id))
        self.recurrent = recurrent 
        self.unique_id = unique_id

    def __str__(self):
        str_rep = super().__str__() + " ID: {} RECURRENT?: {}"
        return str_rep.format(self.unique_id, self.recurrent) 
    
    def is_similar(self, node):
        if (not (isinstance(node, Jumper))):
            return False
        similar = (self.recurrent == node.recurrent)
        return similar
    
class Genome(list):
    """
    Genome using the Common Genetic Encoding (CGE) from EANT.
    Represents an individual in the population. The Genome
    is represented as a list of nodes, where any node can
    describe a Neuron, Input, or Jumper connection (forward or
    recurrent). The ordering of the nodes is pivotal and
    describes the shape of the neural network encoded by
    this genotype. 
    
    For this implementation, an entry in the list can be
    viewed as a Tuple representing (Node, weight). This allows
    Nodes to be unique. 
    """
    def __init__(self, content):
        list.__init__(self, content)
        
        if (isinstance(content, Genome)):
            # Track the exact indices where mutation happened, for precise CMA
            self.indices_added = content.indices_added
            self.indices_lost = content.indices_lost
            
            self.parent_cluster = content.parent_cluster
            self.protection_left = content.protection_left
            self.neuron_list = content.neuron_list.copy()
            self.input_list = content.input_list.copy()
            self.node_list = content.node_list.copy()
            self.node_id_list = content.node_id_list.copy()
            self.max_id = content.max_id
        else:
            # TODO Can we move these out of Genome?
            # Params for the higher-level EANT algorithm
            self.parent_cluster = None
            self.protection_left = 0 
            # Track the exact indices where mutation happened, for precise CMA
            self.indices_added = []
            self.indices_lost = []
            
            self.neuron_list = []
            self.input_list = []
            self.node_list = []
            self.node_id_list = []
            self.max_id = 0
            # Create lists and do some sanity checks
            for (node, weight) in content:
                if (not (node in self.node_list)):
                    self.node_list.append(node)
                    if (isinstance(node, Neuron)):
                        if (node.unique_id in self.node_id_list):
                            print("ERROR DUPLICATE IDs")
                        else:
                            if (node.unique_id > self.max_id):
                                self.max_id = node.unique_id
                                
                            self.neuron_list.append(node)
                            self.node_id_list.append(node.unique_id)
                    elif (isinstance(node, InputNode)):
                        if (not (node in self.input_list)):
                            self.input_list.append(node)   
      
                     
    def clone(self):
        g = Genome(self.copy())
        g.input_list = self.input_list.copy()
        g.neuron_list = self.neuron_list.copy()
        g.node_list = self.node_list.copy()
        g.node_id_list = self.node_id_list.copy()
        g.max_id = self.max_id
        g.parent_cluster = self.parent_cluster
        g.protection_left = self.protection_left
        
        return g
    
    def step_protection(self):
        if (self.protection_left > 0):
            self.protection_left -= 1
    
    def is_protected(self):
        return self.protection_left > 0
    
    def set_protection(self, level):
        self.protection_left = level
    
    def get_weights(self):
        """
        Get a list with just the weight values used in this CGE Genome. 

        Returns
        -------
        list
            List of weights corresponding to each node in the CGE representation.
            Order matches that of the nodes in the encoding.

        """
        return [x[1] for x in self]
    
    def set_weights(self, weights):
        """
        Set new weights for this individual. Given list of weights must match
        length of this individual. One weight for each node. This method
        will mutate this object.

        Parameters
        ----------
        weights : List
            List of weights corresponding to each node in the CGE representation.

        Returns
        -------
        None.

        """
        assert(len(weights) == len(self))
        for i, w in enumerate(weights):
            # Each entry in self is a tuple (node, weight)
            node = self[i][0]
            self[i] = (node, w)
                     
                        
    def _set_inputs(self, **inputs):
        """
        Set values for the input nodes. Required for `evaluate` to return
        a valid result. 

        Parameters
        ----------
        **inputs : Dictionary containing values to apply to InputNodes. 
                   Keys must match names of InputNodes. 

        Returns
        -------
        None.
        """
        for name in inputs:
            for input_node in self.input_list:
                if input_node.name == name:
                    input_node.set_value(inputs[name])

    def evaluate(self, start_index=None, stop_index=0, step=False, **inputs):
        """
        Evaluate the Genome in-place. Accomplished by reading
        the Genome right-to-left and performing computations 
        at each respective Nodes. Possible because of linear 
        ordering and stored computation of nodes. 

        Parameters
        ----------
        start_index : Integer representing the first index to evaluate. 
                      Important to remember evaluation is right-to-left. Default
                      is the final index.
        stop_index : Integer representing the final index to evaluate. Important
                     to remember evaluation is right-to-left. Default is 0.
        step : Boolean indicating whether this is the last evaluation of this 
               timestep. In other words, should the genome take a step?
               The default is True.
        **inputs : Dictionary containing values to apply to InputNodes. 
                   Keys must match names of InputNodes. 

        Returns
        -------
        results : List containing a floating point result for each expected 
                  output. 
        """
        self._set_inputs(**inputs)

        # Calculate start and end indicies if given
        start_index = len(self)-1 if start_index == None else start_index
        i = start_index
        
        call_stack = deque([])
        #print ("Evaluating genome: ", str(self))
        # Move right to left in computation
        while (i >= stop_index):
            #print ("Evaluating element ", i)
            # Element in this list is a tuple (node, weight)
            curr_node, curr_weight = self[i]
            
            # Node is a neuron: pop n values with weights from stack.
            # Compute result and push result onto stack.
            # n is Node's arity
            if (isinstance(curr_node, Neuron)):
                #print("Eval a Neuron")
                result = 0
                for k in range(curr_node.arity):
                    val, weight = call_stack.pop()
                    intermediate_result = (weight * val)
                    if (np.isinf(intermediate_result) or np.isneginf(intermediate_result)):
                        print("Neuron result is inf! ")
                        print("val: ", val)
                        print("weight: ", weight)
                        print("Self Genome: ", self)
                        print("Genome[2]", self[1][0])
                        print("Callstack: ", call_stack)
                        exit(1)
                    result = result + intermediate_result

                # Pass dot product result through activation function
                result = curr_node.activate(result)
                call_stack.append((result, curr_weight))
            # Node is an input: push its value and weight on to stack 
            elif (isinstance(curr_node, InputNode)):
                #print("Eval a InputNode")
                call_stack.append((curr_node.value, curr_weight))
            # Node is a Jumper
            else:
                # Node is a Recurrent Jumper: Get the previous output
                # of the node with a matching unique id to this jumper.
                # Push that value on to the stack with this jumper's weight
                if (curr_node.recurrent):
                    #print("Eval a RecurrentJumper")
                    for node in self.neuron_list:
                        if node.unique_id == curr_node.unique_id:
                            #print("Rec Jumper adding to callstack: ", (node.prev_output, curr_weight) )
                            call_stack.append((node.prev_output, curr_weight))
                            break
                        
                # Node is a Forward Jumper: Search for neuron with
                # matching unique_id, and copy its sub-genome. Compute
                # result of sub-genome and push that to stack
                else:
                    #print("Eval a Forward Jumper")
                    found = False
                    
                    # Search for node 
                    for j, (next_node, next_weight) in enumerate(self):
                        # Sub-genome starts at first instance of neuron 
                        # with same unique_id
                        if (isinstance(next_node, Neuron)):
                            if (next_node.unique_id == curr_node.unique_id):
                                found = True
                                break
                    
                    end_index = j
                    if (found):
                        # Copy sub genome
                        constructed = False
                        # Track sub-genome by input/output differences
                        tracking_value = 0
                        while (not constructed):
                            sub_node, sub_weight = self[j]
                            tracking_value += sub_node.tracking_value()
                            j = j+1
                            if tracking_value == 1:
                                constructed = True
                        # Compute sub-genome result and push to stack
                        value = self.evaluate(start_index=j-1, stop_index=end_index, step=False)[0]
                        value = value / next_weight
                        #print("Got subgenome val: ", value)
                        call_stack.append((value, curr_weight))
                    else:
                        print("SUBGENOME ERROR: FAILED TO FIND fJUMPER SOURCE NEURON")
            
            #print("Callstack after iteration ", i, " :", call_stack)
            # Keep moving left
            i -= 1
        
        #print("Got to return")
        # Compile results from remaining stack items
        results = []
        for x in range(len(call_stack)):
            final_val, final_weight = call_stack.pop()
            final_result = final_val * final_weight
            if (np.isinf(final_result) or np.isneginf(final_result)):
                print("What the heck, were getting inf")
                print("Final val: ", final_val)
                print("Final weight: ", final_weight)
            results.append(final_result)
            
        # Finally, update all nodes to track outputs for next timestep
        if step:
            for node in self.node_list:
                node.step()
        return results
        
# --------------------- CGE Generation ---------------------- #
def generate(outputs=1, guided=False, classifier=True, *input_names):
    return _generate_minimal(outputs, guided, classifier, *input_names)

def _generate_minimal(outputs=1, guided=False, classifier=True, *input_names):
    """
    Generates an EANT individual according to the Common Genetic Encoding (CGE).
    Creates the most minimal individual possible: a NN with connections
    from each input node to each output node. Input nodes must be provided names
    for differentiation, and all other neuron nodes will be automatically 
    named. 
    """
    assert(outputs > 0)
    genome = []
    input_nodes = []
    # Make a unique InputNode for each input, according to provided names
    for in_name in input_names:
        input_nodes.append(InputNode(str(in_name)))
    # For each desired output neuron, make the most basic NN: connections
    # from each input to each output neuron. Fully connected with no hidden
    # layers.
    if classifier:
        func = 'sigmoid'
    else:
        func = 'relu'
    for i in range(outputs):
        if guided:
            node = Neuron(i, len(input_names), fn=func) #           
        else:
            node = Neuron(i, 0, fn=func)
        weight = gen_weight()
        genome.append((node, weight))
        # Guided: add fully connected inputs
        if guided:
            for in_node in input_nodes:
                weight = gen_weight()
                genome.append((in_node, weight))
    g = Genome(genome)
    if not guided:
        g.input_list = input_nodes
        for in_node in input_nodes:
            g.node_list.append(in_node)
    # print("Made a genome with input_list: ", g.input_list)
    return g
        
    
def gen_weight():
    return random.uniform(-2, 2)
# --------------------- EANT Mutation ----------------------- #

def mutate_cge(individual, ind_pb=1): 
    if (isinstance(individual, Genome)):
        return _mut_genome(individual, ind_pb)
    elif (isinstance(individual, list)):
        # Assumes list is a valid Genome representation 
        return _mut_genome(Genome(individual), ind_pb)
    else: 
        return NotImplemented

def _mut_add_jumper(individual, idx):
    #print("MUT Add a Jumper at idx", idx)
    # Add Jumper connection
    """ TODO This is the simplest implementation, unclear whether
             this matches original paper. Here, I am randomly
             choosing a source node for a jumper. The positioning
             of the source node relative to the target node
             automatically determines if the added jumper is
             Recurrent or Forward
    """
    new_weight = gen_weight()
    # Find a "from" node for new connection
    from_node = random.choice(individual.neuron_list)
    # target_node is node were adding connection to
    target_node = individual[idx][0] 
    # Jumper is Recurrent if target is before the source node in NN
    # Forward if target is after the source node in NN
    recurrent = (from_node.depth <= target_node.depth)
    # Forward connections cannot be allowed to self-connect
    if not recurrent:
        if from_node is target_node:
            return False
    new_node = Jumper(from_node.unique_id, recurrent=recurrent)
    individual.insert(idx + 1, (new_node, new_weight))
    individual.node_list.append(new_node)
    target_node.arity += 1
    individual.indices_added.append(idx + 1)
    return True
    
def _mut_del_jumper(individual, idx):
    #print("MUT Remove connection at idx", idx)
    # Remove connection
    # Randomly remove any of node's jumper connections
    target_node = individual[idx][0]
    
    i = 1 # Tracks the extra nodes we need to skip (sub networks)
    for j in range(target_node.arity):
        sub_node = individual[idx + j + i][0]
        # Only remove jumpers
        if isinstance(sub_node, Jumper):
            removed = individual.pop(idx + j + i)
            target_node.arity -= 1
            individual.indices_lost.append(idx + j + i)
            return True
        elif isinstance(sub_node, Neuron):
            # Traverse sub genome
            constructed = False
            # Track sub-genome by input/output differences
            tracking_value = 0
            while (not constructed):
                sub_node, _ = individual[idx + j + i]
                tracking_value += sub_node.tracking_value()
                i += 1
                if tracking_value == 1:
                    i -= 1
                    constructed = True
    return False

def _mut_add_subgenome(individual, idx): 
    #print("MUT Add a sub-genome at idx ", idx)
    new_nodes = 0
    # First, add a new Neuron at this location
    new_weight = gen_weight()
    # target_node is node were adding connection to
    target_node = individual[idx][0]
    # Add a new Neuron whose unique_id is 1 above the last largest 
    # Set arity to a random value between 1 and the number of inputs in this NN
    print("# inputs: ", len(individual.input_list))
    new_node = Neuron(individual.max_id + 1, arity=random.randint(1, len(individual.input_list)), depth=target_node.depth + 1)
    individual.max_id += 1
    individual.insert(idx + 1, (new_node, new_weight))
    individual.indices_added.append(idx + 1)
    individual.neuron_list.append(new_node)
    individual.node_list.append(new_node)
    target_node.arity += 1
    new_nodes += 1
    idx += 1
    # Add connections to the new neuron, to form a sub-genome 
    new_target = new_node
    for i in range(new_target.arity):
        need_input = True
        if (random.random() <= 0.5):
            """ Only allowing forward jumpers because of CGE paper """
            # Add a forward jumper connection
            # Find a "from" node for new connection
            forw_choices = [x for x in individual.neuron_list if (x.depth > new_target.depth and x != new_target)]
            if (len(forw_choices) != 0):
                from_node = random.choice(forw_choices) # individual.neuron_list)
                new_weight = gen_weight()
                new_node = Jumper(from_node.unique_id, recurrent=False) # (from_node.depth < new_target.depth))
                need_input = False
        
        # Adding extra if to ensure we always add an input (even if forward_jumper failed)
        if (need_input):
            # Add an input connection
            new_node = random.choice(individual.input_list)
            new_weight = gen_weight()

        # Insert new connection
        individual.insert(idx + 1, (new_node, new_weight))
        individual.indices_added.append(idx + 1)
        individual.node_list.append(new_node)
        new_nodes += 1
        idx += 1
    return new_nodes

def _mut_genome(individual, ind_pb=1):
    new_individual = individual.clone() 
    # Start of new mutation, erase old tracking to start new
    new_individual.indices_added = []
    new_individual.indices_lost = []
    nodes_added = 0
    for i in range(len(individual)):
        node, weight = new_individual[i + nodes_added]
        # Structural mutation only applies to Neurons
        if (isinstance(node, Neuron)):
            # Test if mutation should occur according to ind_pb
            if (random.random() < ind_pb):
                # Mutation will occur
                # Randomly choose type of mutation
                mut_type_prob = random.random()
                if (mut_type_prob <= 0.333):
                    print("add Jumper")
                    if _mut_add_jumper(new_individual, i + nodes_added):
                        nodes_added += 1
                elif (mut_type_prob <= 0.666):
                    print("Add nodes")
                    nodes_added += _mut_add_subgenome(new_individual, i + nodes_added)
                else:
                    print("del jumper")
                    if _mut_del_jumper(new_individual, i + nodes_added):
                        nodes_added -= 1

    return new_individual

# -------- EANT Individual Similarity (Isomorphism) ---------- #
def _subnetwork_repr(genome):
    """
    Creates an adjacency or "subnetwork" representation in a Dictionary.
    Each Key in the dictionary will be a unique Neuron in the Genome. The
    value associated with that Key is a list, representing its subnetwork
    (including itself). 

    Used as a helper method when checking for graph similarity. See 
    `are_graphs_similar()`.
    """
    #print("Got genome for subnetworking: ", genome)
    subnetworks = {}
    for idx, (node, weight) in enumerate(genome):
        if (isinstance(node, Neuron)):
            # Find its subnetwork, add it to the dictionary
            network = []
            
            j = idx
            # Copy sub genome
            constructed = False
            # Track sub-genome by input/output differences
            tracking_value = 0
            while (not constructed):
                sub_node, sub_weight = genome[j]
                tracking_value += sub_node.tracking_value()
                network.append(sub_node)
                j = j+1
                if tracking_value == 1:
                    constructed = True
            
            subnetworks[node] = network
            
    return subnetworks

def _id_repr(genome):
    """
    Creates a dictionary representation mapping Nueron unique id to the 
    Neuron object itself.
    """
    id_dict = {}
    for neuron in genome.neuron_list:
        id_dict[neuron.unique_id] = neuron
        
    return id_dict

def _are_subnetworks_similar(sub_net_1, sub_net_2, id_dict_1, id_dict_2):
    """
    Check if two subnetworks are similar. 

    Parameters
    ----------
    sub_net_1 : List
        List representation of first subnetwork
    sub_net_2 : List
        List representation of second subnetwork
    id_dict_1 : Dicitonary
        {node id : node} mapping for all Neurons in first parent Genome
    id_dict_2 : Dicitonary
        {node id : node} mapping for all Neurons in second parent Genome

    Returns
    -------
    True if subnetworks are similar, else False.
    """
    i = 1 # Skip first Neuron
    used_indices = [] # Nodes in second subnetwork already used in a match
    while (i < len(sub_net_1)):
        node_to_check = sub_net_1[i]
        # Search for a similar node in second subnetwork
        j = 1
        sub_node = None # Sanity 
        found = False
        while (j < len(sub_net_2)):
            sub_node = sub_net_2[j]
            if (j not in used_indices):
                # TODO This logic is redundant 
                # Simplest case: InputNodes
                if isinstance(node_to_check, InputNode):
                    if node_to_check.is_similar(sub_node):
                        used_indices.append(j)
                        found = True
                        break
                # Edge cases with Jumpers
                elif isinstance(node_to_check, Jumper):
                    if node_to_check.is_similar(sub_node): 
                        parent_1 = id_dict_1[node_to_check.unique_id]
                        parent_2 = id_dict_2[sub_node.unique_id]
                        # Jumpers are similar if they connect to similar nodes
                        if parent_1.is_similar(parent_2):
                            used_indices.append(j)
                            found = True
                            break
                    elif isinstance(sub_node, Neuron): # Edge case: Jumpers can match with Neuron
                        parent_1 = id_dict_1[node_to_check.unique_id]
                        parent_2 = sub_node
                        # Jumpers are similar if they connect to similar nodes
                        if parent_1.is_similar(parent_2):
                            used_indices.append(j)
                            found = True
                            break
                # Edge cases with Neurons
                elif isinstance(node_to_check, Neuron):
                    if node_to_check.is_similar(sub_node):
                        used_indices.append(j)
                        found = True
                        break
                    elif isinstance(sub_node, Jumper):
                        parent_1 = node_to_check
                        parent_2 = id_dict_2[sub_node.unique_id]
                        # Jumpers are similar if they connect to similar nodes
                        if parent_1.is_similar(parent_2):
                            used_indices.append(j)
                            found = True
                            break
            else:
                # Don't consider a (sub) sub-network's inputs 
                if isinstance(sub_node, Neuron):
                    # Traverse through sub-sub-network
                    constructed = False
                    # Track sub-genome by input/output differences
                    tracking_value = 0
                    while (not constructed):
                        sub_sub_node = sub_net_2[j]
                        tracking_value += sub_sub_node.tracking_value()
                        j = j+1
                        if tracking_value == 1:
                            j -= 1
                            constructed = True
                            
            j += 1
            
        # Don't consider a (sub) sub-network's inputs 
        if isinstance(node_to_check, Neuron):
            # Traverse through sub-sub-network
            constructed = False
            # Track sub-genome by input/output differences
            tracking_value = 0
            while (not constructed):
                sub_sub_node = sub_net_1[i]
                tracking_value += sub_sub_node.tracking_value()
                i += 1
                if tracking_value == 1:
                    i -= 1
                    constructed = True
        i += 1
        # Final check... if there's no matches, the check failed
        if (not found):
            return False

    return True
        

def are_graphs_similar(ind1, ind2):
    """
    Check if two individuals have structural similarity (isomorphic). 

    Parameters
    ----------
    ind1 : Genome
        First EANT individual to compare for isomorphism
    ind2 : TYPE
        Second EANT individual to compare for isomorphism

    Returns
    -------
    True if the two individuals are isomorphic, else False
    """
    # Individuals must have same number of nodes (and weights)
    if (len(ind1) != len(ind2)):
        return False
    # Create subnetwork dictionaries to save computation
    sub_repr_1 = _subnetwork_repr(ind1)
    sub_repr_2 = _subnetwork_repr(ind2)
    # Create neuron ID lookups to save computation
    id_dict_1 = _id_repr(ind1)
    id_dict_2 = _id_repr(ind2)
    # Create list copies to allow modifications for ease of tracking
    nodes_by_depth_1 = ind1.neuron_list.copy()
    nodes_by_depth_2 = ind2.neuron_list.copy()
    # Sort by depth for convenience 
    nodes_by_depth_1.sort(key=lambda x: x.depth)
    nodes_by_depth_2.sort(key=lambda x: x.depth)
    # For every neuron in our first graph, check for a similar node
    # somewhere in second graph
    for node in nodes_by_depth_1:
        # Try to find a similar node in ind2
        i = 0
        found_match = False
        # Using while loop so we can mutate second list 
        while (i < len(nodes_by_depth_2)):
            node_2 = nodes_by_depth_2[i]
            # Check if parent node is similar
            if node.is_similar(node_2):
                # Check if all nodes in subnetworks appear similar
                # This is a loose check... does not check Jumpers go to identical nodes (just similar)
                if _are_subnetworks_similar(sub_repr_1[node], sub_repr_2[node_2], id_dict_1, id_dict_2): 
                    # This node and subnetwork is similar. Remove it from
                    # potential candidates of future matches 
                    nodes_by_depth_2.remove(node_2)
                    found_match = True
                    break
                else:
                    return False
            i += 1
        # Found no match for this node. Graphs cannot be similar 
        if (not found_match):
            return False
        
    return True           

# --------------- EANT Graph Isomorphism Testing ------------- #

if __name__ ==  '__main__':
    # Starting trivial: Fully connected networks with no hidden units
    g1 = generate(3, 'x', 'y')
    g2 = generate(3, 'x', 'y')
    assert(are_graphs_similar(g1, g2)) 
    g1 = generate(7, 'x', 'y')
    g2 = generate(7, 'x', 'y')
    assert(are_graphs_similar(g1, g2)) 
    g1 = generate(2, 'x', 'y')
    g2 = generate(7, 'x', 'y')
    assert(not are_graphs_similar(g1, g2)) 

    # Adding complexity: hidden layers, but no Jumpers
    g1 = []
    g1.append((Neuron(0, 3), 1))
    g1.append((InputNode("z"), 1))
    g1.append((Neuron(1, 2), 1))
    g1.append((InputNode("x"), 1))
    g1.append((InputNode("y"), 1))
    g1.append((Neuron(2, 2), 1))
    g1.append((InputNode("x"), 1))
    g1.append((InputNode("y"), 1))
    g1 = Genome(g1)

    g2 = []
    g2.append((Neuron(7, 3), 1))
    g2.append((Neuron(8, 2), 1))
    g2.append((InputNode("y"), 1))
    g2.append((InputNode("x"), 1))
    g2.append((Neuron(9, 2), 1))
    g2.append((InputNode("x"), 1))
    g2.append((InputNode("y"), 1))
    g2.append((InputNode("z"), 1))
    g2 = Genome(g2)
    assert(are_graphs_similar(g1, g2))

    g2 = []
    g2.append((Neuron(7, 3), 1))
    g2.append((Neuron(8, 2), 1))
    g2.append((InputNode("x"), 1))
    g2.append((InputNode("y"), 1))
    g2.append((Neuron(9, 2), 1))
    g2.append((InputNode("y"), 1))
    g2.append((InputNode("y"), 1))
    g2.append((InputNode("z"), 1))
    g2 = Genome(g2)
    assert(not are_graphs_similar(g2, g1))

    # Adding complexity again: allowing Jumpers
    g1 = []
    g1.append((Neuron(0, 2, depth=0), 1))
    g1.append((Neuron(1, 1, depth=1), 1))
    g1.append((Neuron(2, 2, depth=2), 1))
    g1.append((InputNode("x"), 1))
    g1.append((InputNode("y"), 1))
    g1.append((Neuron(3, 1, depth=1), 1))
    g1.append((Jumper(2), 1))

    g1 = Genome(g1)

    g2 = []
    g2.append((Neuron(4, 2, depth=0), 1))
    g2.append((Neuron(5, 1, depth=1), 1))
    g2.append((Jumper(7), 1))
    g2.append((Neuron(6, 1, depth=1), 1))
    g2.append((Neuron(7, 2, depth=2), 1))
    g2.append((InputNode("x"), 1))
    g2.append((InputNode("y"), 1))
    g2 = Genome(g2)
    assert(are_graphs_similar(g2, g1))

    g2 = []
    g2.append((Neuron(4, 2, depth=0), 1))
    g2.append((Jumper(7), 1))
    g2.append((Neuron(6, 1, depth=1), 1))
    g2.append((Neuron(7, 2, depth=2), 1))
    g2.append((InputNode("x"), 1))
    g2.append((InputNode("y"), 1))
    g2 = Genome(g2)
    assert(not are_graphs_similar(g2, g1))

    g1 = []
    g1.append((Neuron(0, 1, depth=0), 1))
    g1.append((Neuron(1, 2, depth=1), 1))
    g1.append((InputNode("x"), 1))
    g1.append((InputNode("y"), 1))
    g1.append((Neuron(2, 2, depth=0), 1))
    g1.append((Jumper(1), 1))
    g1.append((Neuron(3, 1, depth=1), 1))
    g1.append((Jumper(1), 1))

    g1 = Genome(g1)

    g2 = []
    g2.append((Neuron(0, 1, depth=0), 1))
    g2.append((Neuron(1, 1, depth=1), 1))
    g2.append((Jumper(3), 1))
    g2.append((Neuron(2, 2, depth=0), 1))
    g2.append((Jumper(1), 1))
    g2.append((Neuron(3, 2, depth=1), 1))
    g2.append((InputNode("x"), 1))
    g2.append((InputNode("y"), 1))
    g2 = Genome(g2)
    assert(not are_graphs_similar(g2, g1))

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
    assert(are_graphs_similar(g2, g1))

    # TODO Probably still missing some bug exposing edge-cases
    
    # -------------------- Sandbox EANT Testing ------------------ #
    genome = []
    neuron_0 = Neuron(0, 2)
    neuron_1 = Neuron(1, 2)
    neuron_2 = Neuron(2, 4)
    neuron_3 = Neuron(3, 2)
    input_x = InputNode("x")
    input_y = InputNode("y")
    jumper_0 = Jumper(0, recurrent=True)
    jumper_1 = Jumper(3)

    genome.append((neuron_0, 0.6))
    genome.append((neuron_1, 0.8))
    genome.append((neuron_3, 0.9))
    genome.append((input_x, 0.1))
    genome.append((input_y, 0.4))
    genome.append((input_y, 0.5))
    genome.append((neuron_2, 0.2))
    genome.append((jumper_1, 0.3))
    genome.append((input_x, 0.7))
    genome.append((input_y, 0.8))
    genome.append((jumper_0, 0.2))

    test_genome = Genome(genome)
    print(test_genome.evaluate(x=1, y=1))

    for node, _ in genome:
        print(node)
        
    generated_genome = generate(5, "x", "y")
    print(generated_genome)
    for (node, weight) in generated_genome:
        print("[N= {} , W= {} ]".format(node, weight))
        
    mutated_genome = generated_genome
    for i in range(25):
        mutated_genome = mutate_cge(mutated_genome, ind_pb=0.05)
        print("MUTATION RESULTS:")
        for (node, weight) in mutated_genome:
            print("[N= {} , W= {} ]".format(node, weight))
            
    print("Final computation: ", mutated_genome.evaluate(x=1, y=1))