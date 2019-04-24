from __future__ import print_function
import json
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from os import listdir
from os.path import isfile, join

PATH = "C:/Users/sneha/Desktop/IRProject/bbcsport/dataset/data"

categories=None

def getKeywords():
    keyList = []
    fileList = getFiles()

    for file in fileList:
        f=(open(PATH + "/" + file).read())
        keyList.append(f)

    return keyList

def getFiles():
    onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
    return onlyfiles

def createClusters():
    dataset=getKeywords()
    true_k = 5
    vectorizer = TfidfVectorizer(max_df=0.5,min_df=2, stop_words='english',use_idf=True)
    X = vectorizer.fit_transform(dataset)
    svd = TruncatedSVD(100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1100, n_init=100, verbose=True)
    fit = km.fit(X)
    return km


def givenames(namelist):
    ans = []
    km = createClusters()
    for doc, clusterno in zip(getFiles(),km.labels_):
        if doc in namelist:
            ans.append(clusterno)
    return ans


def clusterindex():
    clusterdic = {}
    km = createClusters()
    for doc, clusterno in zip(getFiles(),km.labels_):
        location=clusterdic.setdefault(str(clusterno),[])
        location.append(doc)
    writeIndex(clusterdic)
    return clusterdic


def writeIndex(index):
    json.dump(index, open("C:/Users/sneha/Desktop/IRProject/clusterindex.txt", 'w'))

if __name__=="__main__":
    writeIndex(clusterindex())