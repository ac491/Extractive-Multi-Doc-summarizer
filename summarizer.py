import nltk
from extractor import Extractor
import math
import re
import os
from os import listdir
from os.path import isfile, join
import json 

N = 5
PATH = "C:/Users/sneha/Desktop/IRProject/bbcsport/dataset/data"

class Summarizer:
    
    def getFiles(self):
        onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
        return onlyfiles

    def getDocTokens(self, document):
        tokens = set()
        e = Extractor(document)
        tokens = set(e.rank_words())
        return tokens

    def countFreq(self, document, word):
        return sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), document.lower()))

    def getDocTF(self, document, token):
        freq = self.countFreq(document, token)
        tf = 1 + math.log10(freq) if freq != 0 else 0
        return tf

    def calculateIDF(self, corpus, tokens):
        sentences = nltk.sent_tokenize(corpus)
        n = len(sentences)
        idf = {}

        for token in tokens:
            idf[token] = 0

        for token in tokens:
            for sentence in sentences:
                if token in sentence:
                    if token not in idf:
                        idf[token] = 0 
                    idf[token] += 1

        for token in tokens:
            df = idf[token]
            if df != 0:
                idf[token] = math.log10(n/df)
            else:
                idf[token] = 0

        return idf
        
    def vectorizeDoc(self, document, tokens, idf):
        tfIdf = []
        for token in tokens:
            tfIdf.append(self.getDocTF(document, token) * idf[token])
        return tfIdf

    def normalize(self, docWeights):
        normWeights = docWeights
        if all(i == 0 for i in normWeights):
            return normWeights
        squared = [i ** 2 for i in docWeights]
        normWeights = [float(i)/float(math.sqrt(sum(squared))) for i in docWeights]
        return normWeights

    def computeCosine(self, document, tokens, idf, corpus):
        cosine = 0
        docVector = self.normalize(self.vectorizeDoc(document.lower(), tokens, idf))
        for doc in corpus:
            if doc != document:
                vector = self.normalize(self.vectorizeDoc(doc.lower(), tokens, idf))
                result = [a*b for a,b in zip(vector, docVector)]
                cosine += sum(result)
        return cosine

    def getSummary(self, tokens, idf, corpus):
        weights = []
        for doc in corpus:
            dict = {}
            cosine = self.computeCosine(doc.lower(), tokens, idf, corpus)
            dict['doc'] = doc
            dict['weight'] = cosine
            weights.append(dict)
        result = sorted(weights, key=lambda k: k['weight'])
        return result[0:N]

    def writeIndex(self, index):
        json.dump(index, open("C:/Users/sneha/Desktop/IRProject/summary_index.txt", 'w'))   


    def createSummaryIndex(self):
        index = {}
        for file in self.getFiles():
            tokens = self.getDocTokens((open(PATH + "/" + file).read()).lower())
            idf = self.calculateIDF((open(PATH + "/" + file).read()).lower(), tokens)
            sentences = nltk.sent_tokenize(open(PATH + "/" + file).read())
            first = sentences.pop(0)
            summary = self.getSummary(tokens, idf, sentences)
            result = first
            for i in range(len(summary)):
                result = result + " " + summary[i].get('doc')
            result = result.replace('\n', ' ')
            index[file] = result
        self.writeIndex(index)


if __name__=="__main__":
    s = Summarizer()
    s.createSummaryIndex()
    