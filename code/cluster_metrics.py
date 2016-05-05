from sklearn import metrics
from scipy.stats import entropy
from text_clustering import *
import numpy as np


def label_entropy(labels):
    probs = np.unique(labels,return_counts=True)[1]
    probs = probs/probs.sum()
    return entropy(probs)
    
    
def NMI_by_max(labels,cluster_labels,label_H=None):
    if not label_H:
        label_H = label_entropy(labels)
    
    cluster_H = label_entropy(cluster_labels)
    
    MI = metrics.mutual_info_score(labels,cluster_labels)
    
    return MI/max(label_H,cluster_H)


def NMI_by_mean(labels,cluster_labels,label_H=None):
    if not label_H:
        label_H = label_entropy(labels)
    
    cluster_H = label_entropy(cluster_labels)
    
    MI = metrics.mutual_info_score(labels,cluster_labels)
    
    return 2.0*MI/(label_H + cluster_H)



def label_averaged_score(labels,clust,metric):
    unique_labels, label_counts = np.unique(labels,return_counts=True)
    
    scores = np.empty(len(unique_labels))
    
    for i in range(len(unique_labels)):
        best_score = -1*np.inf
        label_vec = get_label_vec(unique_labels[i],labels)
        
        for cluster_vec in clust.DFS(return_labels=True):
            score = metric(label_vec,cluster_vec)
            if score > best_score:
                best_score = score
                
        scores[i] = best_score
        
    return np.sum((scores*label_counts)/len(labels))


def cluster_averaged_score(labels,clust,metric):
    unique_labels, label_counts = np.unique(labels,return_counts=True)
    
    scores = []#np.zeros(clust.num_leaves)
    cluster_counts = []
    
    for node in clust.DFS(return_labels=False):
        cluster_counts.append(len(node.IDs))
        best_score = 0.0
        cluster_vec = get_cluster_indicator(node.IDs,len(labels))
        
        for i in range(len(unique_labels)):
            label_vec = get_label_vec(unique_labels[i],labels)
            score = metric(label_vec,cluster_vec)
            if score > best_score:
                best_score = score
                
        scores.append(best_score)
    
    scores = np.array(scores)
    return np.sum((scores*cluster_counts)/np.sum(cluster_counts))



def maxMI(labels,clust):
    max_MI = 0.0
    
    for i in range(clust.num_leaves):
        clusters = clust[i]
        cluster_labels = np.zeros(len(labels))
        
        for index,cluster_IDs in enumerate(clusters):
            cluster_labels[cluster_IDs] = index
            
        MI = metrics.mutual_info_score(labels,cluster_labels)
        if MI > max_MI:
            max_MI = MI
            
    return max_MI


def maxNMI(labels,clust):
    max_NMI = 0.0
    
    for i in range(clust.num_leaves):
        clusters = clust[i]
        cluster_labels = np.zeros(len(labels))
        
        for index,cluster_IDs in enumerate(clusters):
            cluster_labels[cluster_IDs] = index
            
        NMI = metrics.normalized_mutual_info_score(labels,cluster_labels)
        if NMI > max_NMI:
            max_NMI = NMI
            
    return max_NMI
    

def maxARI(labels,clust):
    max_ARI = 0.0
    
    for i in range(clust.num_leaves):
        clusters = clust[i]
        cluster_labels = np.zeros(len(labels))
        
        for index,cluster_IDs in enumerate(clusters):
            cluster_labels[cluster_IDs] = index
            
        ARI = metrics.adjusted_rand_score(labels,cluster_labels)
        if ARI > max_ARI:
            max_ARI = ARI
            
    return max_ARI


def get_clustering_score(labels,clusters,score_func):
    cluster_labels = get_cluster_labels(clusters,len(labels))
    
    score = score_func(labels,cluster_labels)
    
    return score

