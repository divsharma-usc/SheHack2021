from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from heapq import nlargest
import operator

# Create your views here.
@api_view(['POST'])
def create_summary(request):
    if request.method == 'POST':
        text = request.data['text']
        
        nltk.download('punkt')
        nltk.download('wordnet')
        
        sentences = sent_tokenize(text)
        stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most']
        stopwords = stop_words + list(punctuation)

        tfidf = TfidfVectorizer(tokenizer=tokenize)
        tfs = tfidf.fit_transform([text])
        freqs = {}
        feature_names = tfidf.get_feature_names()
        for col in tfs.nonzero()[1]:
            freqs[feature_names[col]] = tfs[0, col]

        important_sentences = defaultdict(int)

        for i, sentence in enumerate(sentences):
            for token in word_tokenize(sentence.lower()):
                if token in freqs:
                    important_sentences[i] += freqs[token]
        print(important_sentences.get)
        number_sentences = int(len(sentences) * 0.20)
        index_important_sentences = nlargest(number_sentences,important_sentences,important_sentences.get)

        response = ''
        for i in sorted(index_important_sentences):
            response =  response + sentences[i]

        return Response(response)

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(WordNetLemmatizer().lemmatize(item))
    return stems   