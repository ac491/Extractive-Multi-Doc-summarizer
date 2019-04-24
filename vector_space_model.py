from extractor import Extractor
import os
from os import listdir
from os.path import isfile, join
import re
import json
import math

PATH = "C:/Users/sneha/Desktop/IRProject/bbcsport/dataset/data"

def getFiles():
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    return onlyfiles

def getKeywords():
    keyList = set()
    fileList = getFiles()

    for file in fileList:
        e = Extractor(open(PATH + "/" + file).read())
        keyList |= set(e.rank_words())

    return keyList        

def vector_space_index():
    index = {}
    fileList = getFiles()
    words = getKeywords()
    for word in words:
        location = index.setdefault(word, [])
        subDict = {}
        df = 0
        list = []
        for file in fileList:
            postingList = {}
            if word in (open(PATH + "/" + file).read()).lower():
                pos = postingList.setdefault(file, [])
                freq = countOccurence(file, word)
                freq = 1+ math.log10(freq) if freq != 0 else 0
                pos.append(freq)
                df = df + 1
            if bool(postingList):    
                list.append(postingList)
        subDict['df'] = df
        subDict['postings'] = list
        location.append(subDict)
    return index

def writeIndex(index):
       json.dump(index, open("C:/Users/sneha/Desktop/IRProject/vector_index.txt", 'w'))   

def readIndex():
        index = {}
        index = json.load(open("C:/Users/sneha/Desktop/IRProject/vector_index.txt"))
        return index 

def getIndex():
    index = {}
    if os.stat("C:/Users/sneha/Desktop/IRProject/vector_index.txt").st_size == 0:
        index = vector_space_index()
        writeIndex(index)
    else:
        index = readIndex()

    return index

def countOccurence(file, word):
    return sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), (open(PATH + "/" + file).read()).lower()))    

def getRelevantDocs(tokens, index):
    releveantDocs = set()
    for token in tokens:
        if token in index.keys():
            dict = index[token][0]
            list = dict['postings']
            for posting in list:
                releveantDocs |= posting.keys()
    return releveantDocs            

def countFreq(query, word):
    return sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), query.lower()))

def computeQuery(query, tokens, index):
    weights = []
    for token in tokens:
        if token in index.keys():
            freq = countFreq(query, token)
            tf = 1 + math.log10(freq)
            df = index[token][0]['df']
            idf = math.log10(len(getFiles())/df)
            weights.append(tf*idf)
        else:
            weights.append(0)
    return(weights)

def computeDoc(query, tokens, doc, index):
    weights = []
    for token in tokens:
        if token in index.keys():
            postings = index[token][0]['postings']
            if any(doc in dict for dict in postings):
                d = next(dict for i,dict in enumerate(postings) if doc in dict)
                weights.append(d[doc][0])
            else:
                weights.append(0)
    return weights

def normalize(queryWeights):
    normWeights = queryWeights
    if all(i == 0 for i in normWeights):
        return normWeights
    squared = [i ** 2 for i in queryWeights]
    normWeights = [float(i)/float(math.sqrt(sum(squared))) for i in queryWeights]
    return normWeights

def getDocRelevance(query, index):
    tokens = []
    weights = []
    e = Extractor(query)
    tokens = e.rank_words()
    tokens = set(tokens) - set([""])
    docs = getRelevantDocs(tokens, index)
    queryWeights = normalize(computeQuery(query, tokens, index))
    for doc in docs:
        dict = {}
        dict['name'] = doc
        docWeights = normalize(computeDoc(query, tokens, doc.lower(), index))
        result = [a*b for a,b in zip(queryWeights, docWeights)]
        dict['weight'] = sum(result)
        weights.append(dict)
    return weights

def search(query):
    index = getIndex()
    result = getDocRelevance(query, index)
    result = sorted(result, key=lambda k: k['weight']) 
    if len(result) < 5:
        return result
    else:
        return result[-5:len(result)]        

def getResults(query):
    docs = []
    result = search(query)
    if len(result) == 0:
        print("No results found!")
    else:    
        for i in range(len(result)):
            if result[-(i+1)].get('weight') == 0.0:
                continue
            docs.append(result[-(i+1)].get('name'))
    return docs


if __name__=="__main__":
   getIndex()