#encoding:utf-8

import pandas as pd
import numpy as np
import scipy as scipy
import os
import pickle

# convert sparse dict representation to a dense numpy array with the specified size
def dict_to_np(d,length):
    v = np.zeros((1,length),dtype='float')
    for k,c in d.items():
        v[0,k] = c
    return v

# reduce the indices of the terms to range(0,total_distinct_indices_in_vocab): makes for smaller np arrays
def compress_indices(docs):
    # collect all the indices that actually appear
    indices = {}
    newdocs = []
    i = 0
    for doc in docs:
        newdoc = {}
        for key in doc:
            index = indices.get(key,i)
            newdoc[index] = doc[key]
            if index==i:
                indices[key] = i
                i+=1
        newdocs.append(newdoc)
    return newdocs,indices

class dociter:
    def __init__(self,docs):
        self.docs = docs
        self.i = 0
    
    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= len(self.docs):
            raise StopIteration
        doc = self.docs[self.i]
        self.i += 1
        return doc[2]
        
def sparse_docs_to_array(docs,vocab):
    V=len(vocab)
    # compress the token indices
    newdocs, new_vocab_indices = compress_indices(dociter(docs))
    for i in range(len(docs)):
        docs[i] = (docs[i][0],docs[i][1],newdocs[i])
    
    # build the new vocab with the compressed indices
    newvocab = {}
    for index in new_vocab_indices:
        newvocab[new_vocab_indices[index]] = vocab[index]
    
    # build the numpy array
    docvecs = np.zeros(shape=(len(docs),V),dtype='float')
    for doc in docs:
        docvecs[doc[0],:] = dict_to_np(doc[2],V)
        
    # return the new document vector array and the new vocab
    return docvecs,newvocab
    
def topic_composition_to_array(file_name):
    data =pd.read_csv(file_name,delimiter="\t",header=None)
    data.drop([0,1],axis=1,inplace=True)
    return data.values
