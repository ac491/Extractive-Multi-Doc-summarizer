from summarizer import Summarizer
import re
import os
from os import listdir
from os.path import isfile, join
import json 
import numpy as np
import vector_space_model as vsp
import math
import clustering as c

PATH = "C:/Users/sneha/Desktop/IRProject/bbcsport/dataset/data"

def getDissimilarDocs(relevantDoc, docList, sumIndex, docIndex):
    similarity = []
    relevantSummary = sumIndex[relevantDoc]
    for doc in docList:
        if doc != relevantDoc:
            dict = {}
            summary = sumIndex[doc]
            dict['name'] = doc 
            dict['score'] = getSimilarity(relevantSummary, summary, docIndex)
            similarity.append(dict)
    return sorted(similarity, key=lambda k: k['score'])

def vectorizeDoc(doc, docIndex):
    weights = []
    s = Summarizer()
    for token in vsp.getKeywords():
        if token in docIndex.keys():
            tf = s.getDocTF(doc, token)
            df = docIndex[token][0]['df']
            idf = math.log10(50/df)
            weights.append(tf*idf)
        else:
            weights.append(0)
    weights = s.normalize(weights)
    return weights

def getSimilarity(doc1, doc2, docIndex):
    w1 = vectorizeDoc(doc1, docIndex)
    w2 = vectorizeDoc(doc2, docIndex)
    result = [a*b for a,b in zip(w1, w2)]
    cosine = sum(result)
    return cosine

def readIndex():
    index = {}
    index = json.load(open("C:/Users/sneha/Desktop/IRProject/summary_index.txt"))
    return index

def readClusterIndex():
    index = {}
    index = json.load(open("C:/Users/sneha/Desktop/IRProject/clusterindex.txt"))
    return index 

if __name__=="__main__":
    query = input("Enter search query: ")
    results = vsp.getResults(query)
    clusters = c.givenames(results)
    clusterIndex = readClusterIndex()
    summary = ""
    print(results)
    visited = []
    for i in range(len(results)):
        if clusters[i] not in visited:
            summary += (readIndex())[results[i]]
            docs = clusterIndex[str(clusters[i])]
            similarity = getDissimilarDocs(results[i], docs, readIndex(), vsp.readIndex())
            print(similarity)
            for score in similarity:
                if score['score'] < 0.5 and score['score'] > 0.2:
                    summary += (readIndex())[score['name']]
            visited.append(clusters[i])
    print("\n")
    print(summary)
        

