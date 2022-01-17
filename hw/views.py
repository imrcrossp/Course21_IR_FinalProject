import hw.models as mod
import json
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import gensim
from gensim.models import Word2Vec
import os
from django.conf import settings
import matplotlib.pyplot as plt
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
import pandas as pd
import re
import plotly
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from gensim import corpora
from gensim.summarization import bm25
from gensim.models import Word2Vec
import random
import glob

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

input_word = ""
target_col = []
target_sentence = []
model = Word2Vec.load('hw/static/assets/docs/word2vec.model')
search_text = ""
search_form = ""
# Create your views here.
# @ensure_csrf_cookie

def cal_bm25(input_word, article_row):
	p_stemmer = PorterStemmer()
	for i in range(len(input_word)):
		input_word[i] = input_word[i].split('%')[0]
	#print(input_word)
	global tokenizer
	text = []
	# print(article_row)
	scores = []
	if (len(article_row)>0):
		article_list = []
		for a in article_row:
			a_split = a.replace('?', ' ').replace('(', ' ').replace(')', ' ').split(' ')
			# 詞干提取
			stemmed_tokens = [p_stemmer.stem(i) for i in a_split]
			article_list.append(stemmed_tokens)

		query = input_word
		query_stemmed = [p_stemmer.stem(i) for i in query]

		# bm25模型
		bm25Model = bm25.BM25(article_list)
		# 逆文件頻率
		average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
		scores = bm25Model.get_scores(query_stemmed, average_idf)
	return scores

def read_data(index):
	num_per_class = 250
	if (index < num_per_class):
		data = pd.read_csv(os.path.join(settings.BASE_DIR, 'hw/static/assets/docs/AKI/' + str(index // 10 + 1) + '.csv'), encoding='utf-8', engine='python')
	elif (index >= num_per_class and index < 2 * num_per_class):
		data = pd.read_csv(os.path.join(settings.BASE_DIR, 'hw/static/assets/docs/diabetes mellitus/' + str(index // 10 + 1 - 25) + '.csv'), encoding='utf-8', engine='python')
	elif (index >= 2 * num_per_class and index < 3 * num_per_class):
		data = pd.read_csv(os.path.join(settings.BASE_DIR, 'hw/static/assets/docs/heart disease/' + str(index // 10 + 1 - 50) + '.csv'), encoding='utf-8', engine='python')
	else:
		data = pd.read_csv(os.path.join(settings.BASE_DIR,'hw/static/assets/docs/covid19/' + str(index // 10 + 1 - 75) + '.csv'),encoding='utf-8', engine='python')
	return data

def init_search():
	BASE_PATH = os.getcwd()
	idx = 750
	global target_sentence
	target_sentence = []
	counter = 0
	filenames = glob.glob(BASE_PATH+"/hw/static/assets/docs/covid19/*.csv")
	filenames = sorted(filenames, key = lambda x: int(x.split('/')[-1].split('.')[0]))
	
	for fn in filenames:
		with open(fn, newline='') as csvfile:
			df = pd.read_csv(csvfile)
			for index, row in df.iterrows():
				if not pd.isna(row['title']) and not pd.isna(row['abstract']) > 0:
					file_abstract = row['abstract']
					file_title = row['title']
					if len(file_abstract) > 200:
						target_sentence.append([0, idx, file_title, file_abstract, file_abstract[:200]+' ...', 0, 'COVID 19'])
					else:
						target_sentence.append([0, idx, file_title, file_abstract, file_abstract, 0, 'COVID 19'])
					if counter == 10:
						break
					else:
						counter = counter + 1
				idx = idx + 1
			if counter == 10:
				break
	return target_sentence[:10]

def search(tf_form, text, _type):
	global input_word
	global target_col
	global target_sentence
	ps = PorterStemmer()

	if(text!='none'):
		input_word = text.split(' ')
		for i in range(len(input_word)):
			input_word[i] = ps.stem(input_word[i])

	tfidf = pd.read_csv(os.path.join(settings.BASE_DIR, 'hw/static/assets/docs/tfidf (2).csv'), encoding='utf-8', engine='python')
	#print(input_word)
	target_col = []
	for word in input_word:
		for i in list(tfidf.columns):
			if(ps.stem(i.split('%')[0])==word):
				target_col.append(i)
	#print(target_col)

	num_per_class = 250
	target_sentence = []
	target_sentence_bm25 = []
	type_list = []
	for i in range(len(tfidf)):
		find = False
		data = read_data(i)
		score = 0
		for j in target_col:
			score = score + tfidf[j][i]
			if(tfidf[j][i]>0 and not find):
				find = True
				#cal_sentence(i, data['abstract'][i % 10])
				double = False
				if (_type == "aki" and i < num_per_class):
					double = True
				elif (_type == "diabetes_mellitus" and i >= num_per_class and i < 2 * num_per_class):
					double = True
				elif (_type == "heart_disease" and i >= 2 * num_per_class and i < 3 * num_per_class):
					double = True
				elif (_type == "covid19" and i >= 3 * num_per_class and i < 4 * num_per_class):
					double = True
				type_list.append([i // 250, double])
				target_sentence_bm25.append(data['abstract'][i % 10])  # [abstract]
				file_abstract = data['abstract'][i%10]
				if len(file_abstract) > 200:
					target_sentence.append([round(score, 2), i, data['title'][i % 10], file_abstract, file_abstract[:200]+" ..."])
				else:
					target_sentence.append([round(score, 2), i, data['title'][i % 10], file_abstract, file_abstract])


	score_bm25 = cal_bm25(target_col, target_sentence_bm25)
	for i in range(len(target_sentence)):
		if(score_bm25[i]==0):
			score_bm25[i] = 0.1 + random.randint(0, 10)/100.0 # if bm25_score = 0, random(0.1x)

		if(type_list[i][1]):# double by theme
			target_sentence[i][0] = 2 * target_sentence[i][0]
			score_bm25[i] = 2 * score_bm25[i]
		target_sentence[i].append(round(score_bm25[i], 2)) # [tf_score, index, abstract, bm25_score]

		if(type_list[i][0] == 0): target_sentence[i].append('AKI')
		elif (type_list[i][0] == 1): target_sentence[i].append('Diabetes Mellitus')
		elif (type_list[i][0] == 2): target_sentence[i].append('Heart Disease')
		elif (type_list[i][0] == 3): target_sentence[i].append('COVID 19') # [tf_score, index, abstract, bm25_score, type]
		#print(target_sentence[i])

	if(tf_form == 'tfidf'):
		target_sentence = sorted(target_sentence, key = lambda x: x[0],reverse = True)
	else:
		target_sentence = sorted(target_sentence, key = lambda x: x[5], reverse = True)
	return target_sentence[:10]

def rank_article(index):
	global target_sentence

	current_word_embedding = [0] * 250
	word_num = 0
	loss = []

	# calculate current article's word embedding
	data = read_data(index)
	for j in sent_tokenize(data['abstract'][index % 10]):  # j for every sentence
		# calculate word2vec
		for k in word_tokenize(j):  # k for every words
			try:
				current_word_embedding = current_word_embedding + model[k.lower()]
				word_num = word_num + 1
			except:
				continue
	current_word_embedding = [x / word_num for x in current_word_embedding]

	# ||top50 articles word embedding -  current article's word embedding||1
	for i in range(len(target_sentence)):
		data = read_data(target_sentence[i][1])
		word_embedding = [0] * 250
		word_num = 0
		for j in sent_tokenize(data['abstract'][target_sentence[i][1] % 10]):  # j for every sentence
			# calculate word2vec
			for k in word_tokenize(j):  # k for every words
				try:
					word_embedding = word_embedding + model[k.lower()]
					word_num = word_num + 1
				except:
					continue
		word_embedding = [x / word_num for x in word_embedding]
		loss.append(np.linalg.norm((np.array(word_embedding) - np.array(current_word_embedding)), ord=1))

	recommend = []
	loss[np.argmin(loss)] = 100000  # set a big num to delete current article in recommend list
	for i in range(5):
		#print(loss)
		recommend.append(np.argmin(loss))
		loss[np.argmin(loss)] = 100000 # set a big num to delete minimum number
	return recommend




def article(request, index):
	global model
	global input_word
	global target_col
	global target_sentence
	index = int(index)
	recom_article = []
	data = read_data(index)
	text = ""
	if len(target_col) > 0:
		for i in sent_tokenize(data['abstract'][index % 10]):  # i for every sentence
			replaced_s = i
			for j in target_col:
				replaced_s = replaced_s.replace(" " + j.split('%')[0].upper(), " <span style=\"background-color:yellow\">" + j.split('%')[0].upper() + "</span>")
				replaced_s = replaced_s.replace(" " + j.split('%')[0], " <span style=\"background-color:yellow\">" + j.split('%')[0] + "</span>")

				# Replace the title keyword
				insensitive = re.compile(re.escape(j.split('%')[0] + " "), re.IGNORECASE)
				replaced_s = insensitive.sub("<span style=\"background-color:yellow\">" + j.split('%')[0].capitalize() + "</span> ", replaced_s)


			text = text + replaced_s
	else:
		text = data['abstract'][index % 10]
	# random recommend
	recom_index = rank_article(index)
	for i in recom_index:
		recom_data = read_data(target_sentence[i][1])
		recom_article.append([target_sentence[i][1], recom_data['title'][target_sentence[i][1] % 10]])
	return render(request, "hw/article.html", {'title': data['title'][index % 10], 'abstract': text, 'recom_data': recom_article})


def index(request, tf_form = 'tfidf', text = 'tfidf', _type = ""):
	text = request.GET.get('msg')
	tf_form = request.GET.get('selected')
	_type = request.GET.get('_type')
	global search_form
	global search_text
	global target_col
	if _type != None and search_text != "":
		data = search(search_form, search_text, _type)
		return render(request, 'hw/index.html', {'type': 'search', 'docs': data})
	if text == "_init" or (text == "" or text == None):
		data = init_search()
		search_text = ""
		target_col = []
		return render(request, 'hw/index.html', {'type': 'init', 'docs': data})
	else:
		data = search(tf_form, text, _type)
		search_form = tf_form
		search_text = text
		return render(request, 'hw/index.html', {'type': 'search', 'docs': data})
