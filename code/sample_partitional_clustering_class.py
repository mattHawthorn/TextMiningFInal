class BisectingPartitional:
    def __init__(self,criterion,min_leaf=None,max_clusters=None):
        """
        Tree is modeled as a ?
        Leaves need to be easily min/max-indexable for argmin/argmax over objective function of candidate splits (heap?).
        Don't need to be easily loopable since candidate splits are only needed at the most recent 2 leaves? (depends on the objective)
        Nodes need to store 3 things: order of split, indices of contents (datapoints), and pointers to children.
        None of these things change if the 'pointers' to children are actually pointers and not direct name references;
           thus a tuple is viable as a node representation.
        Whole clustering should be indexable by level, for computing metrics over all levels of splitting.
        
        """
        self.root = # whatever a node is []
        self.depth = 0
        self.num_leaves = 0
        self.criterion = criterion
        self.min_leaf = min_leaf
        self.max_clusters = max_clusters
        
        
    def fit(self,data):
        """
        data is assumed to be an indexable of instances exposing a numpy array-like interface.
        """
        # for node in self.leaves:
        #     ...
    
        
    def bestSplit(self,node,n):
        """
        Generate n candidate bisections of a given node.
        """
        # # generate n candidate bisections
        # for iteration in range(n):
        #     # pick 2 random centers from the node datapoints
        #     # assign every datapoint to the closest center
        #     current_split = [[data closest to center 1 by metric],[data closest to center 2 by metric]]
        #
        #     # each time, initialize the objective to the current objective
        #     objective = current_objective
        #     prior_objective = inf
        #
        #     # while the objective keeps improving
        #     while(objective < prior_objective):
        #         prior_objective = objective
        #
        #         # loop over the datapoints in a random order, testing for improvement on each reassignment
        #         suffle(datapoints)
        #         for d in datapoints:
        #             if objective([current_split[0].append(d),current_split[1].remove(d)] <  objective:
        #                 current_split = [current_split[0].append(d),current_split[1].remove(d)]
        #                 objective = objective(current_split)
        #
        #     # finally, if the objective after cycling through the data is better than the best so far,
        #     # store the objective and the split
        #     if objective < best_objective:
        #         best_objective = objective
        #         best_split = current_split
    
    
    def validate(self,labels,weighting):
        """
        Compute a score metric at each level of the clustering (NMI, F1), weighted by some function indicating quality
        (internal such as objective improvement or external in reference to true labels (cheating I think)), and return
        the weighted mean.
        """


        
class node:
    self.children #tuple
    self.docs # list of ids
[objective,split_number,level,child1,child2,list_of_doc_IDs]
        
        