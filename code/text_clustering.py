import numpy as np
from scipy import spatial
from sklearn import cluster
from matplotlib import pyplot as plt
import sparse_to_array
from operator import itemgetter
from heapq import *


def l2norm(v1,v2):
    return np.sum(np.square(v1-v2))

def I2(data, center=None):
    # sum over all rows of cosine(row,center)
    if not center:
        center = data.mean(axis=0)
    return np.float(data.shape[0]) - (data*center).sum()
    
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
        self.children = []
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
            return 0
        
        
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
                if len(node.IDs) > 1:
                    children = self.best_split(node)
                else:
                    continue
                #for child in children:
                #    print(child.IDs)
                # we only save them if their min size is large enough
                if min([len(child.IDs) for child in children]) >= self.min_leaf:
                    # save the new children in the tree
                    node.children = children
                    heappush(leaves,node)
                    
                # if we didn't make it into the if, the new children were not put on the heap
                # and their children were not saved
        
        for node in leaves:
            node.children = []
            
        self.num_leaves = num_leaves
        self.n = self.data.shape[0]
        del self.data
             
    
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
        
        return tuple(nodes)

        
    def center(self,node):
        data = self.data[node.IDs,:]
        return data.mean(axis=0)

        
    def get_clusters(self,n):
        leaves = [self.root]
        
        for i in range(n-1):
            node = heappop(leaves)
            for child in node.children:
                heappush(leaves,child)
        
        return [leaf.IDs for leaf in leaves]

        
    def get_labels(self,n):
        clusters = self.get_clusters(n)
        return get_cluster_labels(clusters,self.n)

    
    def levels(self,n=None,return_labels=True):
        if not n:
            n = self.num_leaves
        
        return LevelIter(self,n,return_labels)
        
        #leaves = [self.root]
        #objective = sum([leaf.objective for leaf in leaves])
        #
        #if return_labels:
        #    labels = np.zeros(self.n,dtype='int')
        #    running_index = 0
        #print(labels)
        #
        #if return_labels:
        #    yield labels, objective
        #else:
        #    yield len(leaves), objective
        # 
        #for i in range(1,n):
        #    node = heappop(leaves)
        #    objective = objective + node.improvement()
        #    
        #    for child in node.children:
        #        heappush(leaves,child)
        #        
        #        if return_labels:
        #            running_index += 1
        #            labels[child.IDs] = running_index
        #            
        #    if return_labels:
        #        yield labels, objective
        #    else:
        #        yield len(leaves), objective
            
    def DFS(self,node=None,return_labels=True):
        if not node:
            node = self.root
        
        if return_labels:
            yield get_cluster_indicator(node.IDs,self.n)
        else:
            yield node
            
        for child in node.children:
            for node in self.DFS(child,return_labels):
                yield node
        

class LevelIter:
    def __init__(self,clust,n=None,return_labels=True):
        self.clust = clust
        self.return_labels = return_labels
        if not n:
            self.n = clust.num_leaves
        else:
            self.n = n
        
        
    def __iter__(self):
        self.i = 0
        self.leaves = [self.clust.root]
        self.objective = sum([leaf.objective for leaf in self.leaves])
        
        if self.return_labels:
            self.labels = np.zeros(self.clust.n,dtype='int')
            self.running_index = 0
            
        return self
        
        
    def __next__(self):
        self.i += 1
        
        if self.i == 1:
            if self.return_labels:
                return self.labels.copy(), self.objective
            else:
                return len(self.leaves), self.objective
        
        if self.i > self.n:
            raise StopIteration
        
        node = heappop(self.leaves)
        self.objective = self.objective + node.improvement()
        
        for child in node.children:
            heappush(self.leaves,child)
            
            if self.return_labels:
                self.running_index += 1
                self.labels[child.IDs] = self.running_index
                
        if self.return_labels:
            return self.labels.copy(), self.objective
        else:
            return len(self.leaves), self.objective




def get_cluster_indicator(IDs,n):
    indicator = np.zeros(n,dtype='int')
    indicator[IDs] = 1
    return indicator
    

def get_label_vec(label,labels):
    label_vec = (labels == label)
    return label_vec.astype('int')
    

def get_cluster_labels(clusters,n=None):
    if not n:
        n = max([np.max(cluster) for cluster in clusters]) + 1
    
    cluster_labels = np.zeros(n)
    
    for index,cluster_IDs in enumerate(clusters):
        cluster_labels[cluster_IDs] = index
    
    return cluster_labels
        
        
#def append_nodes(node,nodes=[]):
#    nodes.append(node)
#    for child in node.children:
#        append_nodes(child,nodes)
#    return
