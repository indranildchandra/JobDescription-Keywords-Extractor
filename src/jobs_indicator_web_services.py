from flask import Flask, jsonify, request

import spacy
import nltk
import random
import pickle
import gensim
import numpy
import pandas as pd

from spacy.lang.en import English
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora, models
from gensim.models import Phrases
from gensim.models import Word2Vec

app = Flask(__name__)
parser = spacy.load('en')

@app.route("/getKeywords", methods=['POST'])
def getKeywords():
	data = request.get_json()['text']
	print("\n------------------------------------------------------------")
	print("Input Data: " + data)
	print("------------------------------------------------------------\n")
	keywords = processKeywords(data)
	print("\n------------------------------------------------------------")
	print("Output Keywords: ")
	print("Noun Phrases: " + str(keywords['noun_phrases']))
	print("BOW Topics: " + str(keywords['bow_topics']))
	print("RF-IDF Topics: " + str(keywords['tfidf_topics']))
	print("------------------------------------------------------------\n")
	
	return jsonify(keywords)

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.is_punct:
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('at')
        elif token.pos_ == "ADJ" or token.pos_ == "VERB" or token.pos_ == "RBR" or token.pos_ == "RBS" or token.pos_ == "RB" or token.pos_ == "RP":
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text(text):
	en_stop = set(nltk.corpus.stopwords.words('english'))
	tokens = tokenize(text)
	tokens = [token for token in tokens if len(token) > 3]
	tokens = [token for token in tokens if token not in en_stop]
	tokens = [get_lemma(token) for token in tokens]
	return tokens

def processKeywords(new_doc):
	keywords =	{
	  "noun_phrases": [],
	  "bow_topics": [],
	  "tfidf_topics": []
	}
	noun_phrases = []
	bow_topics = []
	tfidf_topics = []

	job_description_dictionary = gensim.corpora.Dictionary.load('./../models/job_description_dictionary.gensim')
	job_description_bow_corpus = pickle.load(open('./../models/job_description_bow_corpus.pkl', 'rb'))
	job_description_bow_lda_model = gensim.models.ldamodel.LdaModel.load('./../models/job_description_bow_lda_model.gensim')
	job_description_tfidf_corpus = pickle.load(open('./../models/job_description_tfidf_corpus.pkl', 'rb'))
	job_description_tfidf_lda_model = gensim.models.ldamodel.LdaModel.load('./../models/job_description_tfidf_lda_model.gensim')
	job_description_tfidf_model = models.TfidfModel(job_description_tfidf_corpus)

	document = parser(new_doc)
	new_doc = prepare_text(new_doc)
	new_doc_bow = job_description_dictionary.doc2bow(new_doc)
	print(new_doc_bow)
	new_doc_tfidf = job_description_tfidf_model[new_doc_bow]
	print(new_doc_tfidf)

	print("Noun Phrases...")
	print("--- [Format: Noun Phrase -> Root Text] ---")
	for noun_phrase in document.noun_chunks:
		noun_phrases.append(noun_phrase.text)
		print(noun_phrase.text + " -> " + noun_phrase.root.text)

	print("\n\nTopics relevant to new document are (BoW): ")	

	counter = 0
	for index, score in sorted(job_description_bow_lda_model[new_doc_bow], key=lambda tup: tup[1], reverse=True):
		topic = {
			'topic_index':"",
			'topic_conf':"",
			'topic': ""
		}
		if counter == 0:
			topic['topic_index'] = str(index)
			topic['topic_conf'] = str(score)
			topic['topic'] = job_description_bow_lda_model.print_topic(index, 10)
			bow_topics.append(topic)
			print("Score: {}\t Topic: {}".format(score, job_description_bow_lda_model.print_topic(index, 10)))
			highest_score = score
			counter = counter + 1
		elif highest_score - score <= 0.2:
			topic['topic_index'] = str(index)
			topic['topic_conf'] = str(score)
			topic['topic'] = job_description_bow_lda_model.print_topic(index, 10)
			bow_topics.append(topic)
			print("Score: {}\t Topic: {}".format(score, job_description_bow_lda_model.print_topic(index, 10)))
			counter = counter + 1
		else:
			break
		


	counter = 0
	print("\n\nTopics relevant to new document are (TF-IDF): ")
	for index, score in sorted(job_description_tfidf_lda_model[new_doc_tfidf], key=lambda tup: tup[1], reverse=True):
		topic = {
			'topic_index':"",
			'topic_conf':"",
			'topic': ""
		}
		if counter == 0:
			topic['topic_index'] = str(index)
			topic['topic_conf'] = str(score)
			topic['topic'] = job_description_tfidf_lda_model.print_topic(index, 10)
			tfidf_topics.append(topic)
			print("Score: {}\t Topic: {}".format(score, job_description_tfidf_lda_model.print_topic(index, 10)))
			highest_score = score
			counter = counter + 1
		elif highest_score - score <= 0.2:
			topic['topic_index'] = str(index)
			topic['topic_conf'] = str(score) 
			topic['topic'] = job_description_tfidf_lda_model.print_topic(index, 10)
			tfidf_topics.append(topic)
			print("Score: {}\t Topic: {}".format(score, job_description_tfidf_lda_model.print_topic(index, 10)))
			counter = counter + 1
		else:
			break
		

	keywords['noun_phrases'] = noun_phrases
	keywords['bow_topics'] = bow_topics
	keywords['tfidf_topics'] = tfidf_topics

	return keywords


if __name__ == '__main__':
    app.run(debug=True, port=5000)


# REST API Endpoint Examples
# http://127.0.0.1:5000/getKeywords

# REST API Request Body Examples
# {
# 	"text": "Investment Consulting Company is seeking a Chief Financial Officer. This position manages the company's fiscal and administrative functions, provides highly responsible and technically complex staff assistance to the Executive Director. The work performed requires a high level of technical proficiency in financial management and investment management, as well as management, supervisory and administrative skills." 
# }
# {
# 	"text": "Design right ML & AI algorithms and manage the solution implementation end-to-end Manage client engagements and ensure project delivery as per client expectations. Present solutions in intuitive and effective way to the audience. Actively participate in all activities leading to career progression participating in sales pitches, training in house resources, coaching staff on best practices, client relationship management etc." 
# }
# {
# 	"text": "1.Must have a working knowledge of Asp.net (C#) (both n-tier and MVC must). 2.Must have a working knowledge of Sql Server . 3.Must have a working knowledge of web services, Bootstrap, CSS, Ajax, Java Script and J-query. 4.Report creation . 5.Good Communication skill and convincing power" 
# }