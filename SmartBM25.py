'''
This is an implementation of a smart Key-Word Search Tool using BM25
'''

import os
import pandas as pd
import numpy as np
import pickle
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.models.fasttext import FastText
from rank_bm25 import BM25Okapi
import nmslib
import time
import ftfy

# Load data and create DF
paper=open("/Users/steffen/PycharmProjects/SearchEngine-BM25/text.txt", 'r')
paper_text = paper.read().lower()
#print(paper_text)
# Converting text into List of m rows
text_list = paper_text.split('\n')
#print(text_list)


# Preprocess and tokenise
nlp = spacy.load("en_core_web_sm")
tok_text=[] # for our tokenised corpus
text = [ftfy.fix_text(str(i)) for i in text_list]
#Tokenising using SpaCy:
for doc in tqdm(nlp.pipe(text, disable=["tagger", "parser", "ner"])):
    tok = [t.text for t in doc if (t.is_ascii and not t.is_punct and not t.is_space)]
    tok_text.append(tok)
#print(tok_text)

# FastText
ft_model = FastText(
    sg=1, # use skip-gram: usually gives better results
    #size=100, # embedding dimension (default)
    window=10, # window size: 10 tokens before and 10 tokens after to get wider context
    min_count=5, # only consider tokens with at least n occurrences in the corpus
    negative=15, # negative subsampling: bigger than default to sample negative examples more
    min_n=2, # min character n-gram
    max_n=5 # max character n-gram
)

ft_model.build_vocab(tok_text)

ft_model.train(
    tok_text,
    epochs=6,
    total_examples=ft_model.corpus_count,
    total_words=ft_model.corpus_total_words)

ft_model.save('_fasttext.model')

# Load fasttext and query
ft_model = FastText.load('_fasttext.model')
#print(ft_model)
# Creating BM25 document vectors:
bm25 = BM25Okapi(tok_text)
weighted_doc_vects = []

for i,doc in tqdm(enumerate(tok_text)):
  doc_vector = []
  for word in doc:
    #vector = ft_model[word]
    vector = ft_model.wv[word]
    weight = (bm25.idf[word] * ((bm25.k1 + 1.0)*bm25.doc_freqs[i][word])) / (bm25.k1 * (1.0 - bm25.b + bm25.b *(bm25.doc_len[i]/bm25.avgdl))+bm25.doc_freqs[i][word])
    weighted_vector = vector * weight
    doc_vector.append(weighted_vector)
  doc_vector_mean = np.mean(doc_vector, axis=0)
  weighted_doc_vects.append(doc_vector_mean)
  pickle.dump(weighted_doc_vects, open("weighted_doc_vects.p", "wb"))

#print(weighted_doc_vects)
# Load document vectors, build index and search:
with open("weighted_doc_vects.p", "rb") as f:
    weighted_doc_vects = pickle.load(f)
# create a random matrix to index
data = np.vstack(weighted_doc_vects)

# initialize a new index, using a HNSW(Hierarchical Navigable Small World) index on Cosine Similarity - can take a couple of mins
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex()#{'post': 2})#, print_progress=True)
#print(index)

# querying the index:
input = "shiting assumblay taks oter asembly statns".lower().split(" ")

query = [ft_model.wv[vec] for vec in input]
query = np.mean(query, axis=0)
print(query)
t0 = time.time()
ids, distances = index.knnQuery(query, k=2)
t1 = time.time()
#print(f'Searched {df.shape[0]} records in {round(t1 - t0, 4)} seconds \n')
print(f'Searched {len(text_list)} records in {round(t1 - t0, 4)} seconds \n')
#print(ids, distances)
for i, j in zip(ids, distances):
    print(round(j, 8))
    print(text_list[i])
