import numpy as np
from scipy import spatial
from sklearn import cluster
import sparse_to_array
from heapq import *


def l2norm(v1,v2):
    return np.sum(np.square(v1-v2))

def I2(data, center=None):
    # sum over all rows of cosine(row,center)
    if not center:
        center = data.mean(axis=0)
    return (data*center).sum()
    
def I2Euc(data, center=None):
    # sum over all rows of cosine(row,center)
    if not center:
        center = data.mean(axis=0)
    return ((data-center)**2).sum()

class Node:
    __slots__ = ['children','parent','objective','IDs','center','depth','split_order']
    
    def __init__(self,IDs,parent=None):
        self.parent = parent
        self.IDs = np.array(IDs)
        self.children = None
        self.split_order = None
        self.objective = np.inf
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        
    def __lt__(self,other):
        return self.improvement() < other.improvement()
        
    def improvement(self):
        if self.children:
            return sum([self.children[i].objective for i in range(len(self.children))]) - self.objective
        else:
            return None
        
        
class BisectingPartitional:
    def __init__(self,candidate_splits=10,objective=I2,min_leaf=1,max_leaves=None):
        self.root = None
        self.objective = objective
        #self.metric = metric
        self.candidate_splits = candidate_splits
        self.min_leaf = min_leaf
        self.max_leaves = max_leaves
        
    def fit(self,data):
        self.data = data
        if not self.max_leaves:
            self.max_leaves = self.data.shape[0]
        
        self.root = Node(IDs=list(range(data.shape[0])),parent=None)
        self.root.center = self.center(self.root)
        self.root.objective = self.objective(self.data[self.root.IDs,:])
        self.root.children = self.best_split(self.root)
        leaves = [self.root]
        num_leaves = 1
        
        while len(leaves) > 0:
            # assume all leaves have been assesed for best splits and pop the best one
            node = heappop(leaves)
            node.split_order = num_leaves
            num_leaves +=1
            
            if num_leaves >= self.max_leaves:
                break
                
            # need to assess the children for splits
            for node in node.children:
                # best_split returns 2 node objects
                children = self.best_split(node)
                #for child in children:
                #    print(child.IDs)
                # we only save them if their min size is large enough
                if min([len(child.IDs) for child in children]) > self.min_leaf:
                    # save the new children in the tree
                    node.children = children
                    heappush(leaves,node)
                    
                # if we didn't make it into the if, the new children were not put on the heap
                # and their children were not saved
        
        for node in leaves:
            node.children = None
             
    
    def best_split(self,node):
        #best_objective = np.inf
        #print(node.IDs)
        data = self.data[node.IDs,:]
        #print(data.shape)
        labels = cluster.KMeans(n_clusters=2,n_init = self.candidate_splits).fit_predict(data)
        split = (node.IDs[np.where(labels==0)[0]],node.IDs[np.where(labels==1)[0]])
        #print([len(a) for a in split])
        
        nodes = [Node(IDs=IDs,parent=node) for IDs in split]
        
        for node in nodes:
            node.objective = self.objective(self.data[node.IDs,:])
        
        return nodes
        
    def center(self,node):
        data = self.data[node.IDs,:]
        return data.mean(axis=0)
        
    def __getitem__(self,n):
        leaves = {self.root.split_order:self.root}
        priorities = [self.root.split_order]
        
        for i in range(n-1):
            #print(priorities)
            split_order = heappop(priorities)
            #print(split_order)
            node = leaves[split_order]
            del leaves[split_order]
            for child in node.children:
                if child.split_order:
                    leaves[child.split_order] = child
                    heappush(priorities,child.split_order)
        
        #print(leaves)
        return [leaf.IDs for leaf in leaves.values()]
        
        
        
