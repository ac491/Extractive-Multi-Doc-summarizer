import re, operator, math, string
import nltk

class Extractor:
    def normalize(self,text):
        text = re.sub("[^A-Za-z_]", " ", text)
        text = re.sub(" +"," ",text)
        text = text.lower()
        return text

    def remove_stopwords(self):
        regexSeq = " |\\b".join(self.stopwordsList)
        regexSeq = "\\b"+regexSeq
        text = re.sub(regexSeq,"",self.normalizedText)
        return text

    def __init__(self,text):
        self.stopwordsList = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        self.text = text
        self.normalizedText = self.normalize(text) 
        self.shortText = self.remove_stopwords() 
        self.sentences = [s.lower().strip(string.punctuation) for s in self.text.split(".")]
        


    def extract_words(self,normal):
        words = []
        words = re.split(" +",self.normalizedText) if normal else re.split(" +",self.shortText)
        return words

    def extract_keywords(self):
        keywords = []
        sentences = []
        regexSeq = " |\\b".join(self.stopwordsList)
        regexSeq = "\\b"+regexSeq
        sentences = self.normalizedText.split(".")
        for sentence in sentences:
            temp = re.split(regexSeq,sentence)
            temp = [x.strip() for x in temp if x != ' ' and x != '']
            keywords.append(temp)
        superList = []
        for l in keywords:
            superList = superList + l
        return superList
     

    def rank_words(self):
        keywords = []
        words = []
        wordMap = {}
        keyMap = {}
        keywords = self.extract_keywords()
        words = self.extract_words(False) 
        for word in words:
            freq = self.normalizedText.count(word)
            deg = 0
            for i in range(len(self.sentences)):
                if word in self.sentences[i]:
                    deg += len(self.sentences[i])
            wordScore = deg/freq
            wordMap[word] = wordScore
        for keyword in keywords:
            allWords = re.split(" +",keyword)
            keyScore = 0
            for eachWord in allWords:
                keyScore = keyScore + wordMap[eachWord]
            keyMap[keyword] = keyScore
        sortedMap = sorted(keyMap.items(), key=operator.itemgetter(1))
        sortedMap.reverse()
        return list(wordMap.keys())


