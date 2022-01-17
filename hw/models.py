from django.db import models
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
import nltk
import sqlite3
import pandas as pd
import numpy as np
import os
import math
import string
import ast
import gensim.models
from gensim import utils
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import statistics 
import csv
import glob

def init_index():
	BASE_PATH = os.getcwd()
	counter = 0
	titles = []
	abstracts = []
	filenames = glob.glob(BASE_PATH+"/hw/static/assets/docs/covid19/*.csv")
	for fn in filenames:
		with open(fn, newline='') as csvfile:
			df = pd.read_csv(csvfile)
			for index, row in df.iterrows():
				if not pd.isna(row['title']) and not pd.isna(row['abstract']) > 0:
					titles.append(row['title'])
					if len(row['abstract']) > 70:
						abstracts.append(row['abstract'][:70]+" ...")
					else:
						abstracts.append(row['abstract'])
				if counter == 10:
					break
				else:
					counter = counter + 1
			if counter == 10:
				break
	INFO = zip(titles, abstracts)
	CONTENT = {'docs': INFO, 'titles': titles}
	return CONTENT
	