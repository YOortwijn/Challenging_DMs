"""
| File name: evaluation_kmeans.py
| Author: Yvette Oortwijn
| Email: yvette.oortwijn@gmail.com
| Project: The semantics of meaning
| Date created: 26-10-2020
| Date last modified: 02-11-2020
| Python Version: 3.6
"""

import os
import sys
import semantics
from semantics import SemanticSpace
import evaluate
import context
import random
import re
import itertools
import json
from sklearn.preprocessing import normalize
import numpy as np
import time
import nltk

import gensim
import pandas as pd
from semantics import similarity

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import numpy as np

from collections import defaultdict

termlist_path = "../data/quine_terms.txt"
termlist = []

with open(termlist_path, "r") as f:
    for line in f:
        termlist.extend(line.split())
        
f.close()

cluster_A = ['abstract_singular_term', 'abstract_term', 'adjective', 'article', 'definite_article', 'indefinite_article', 'mass_term', 'demonstrative', 'description', 'general_term', 'singular_term', 'definite_singular_term', 'indefinite_singular_term', 'eternal_sentence', 'indicator_word', 'name', 'noun', 'relative_term', 'substantive', 'observation_sentence', 'occasion_sentence', 'open_sentence', 'pronoun', 'pronominal_singular_term', 'relative_clause', 'relative_pronoun', 'one-word_sentence', 'word', 'verb']
cluster_B = ['abstract_object', 'class', 'concrete_object', 'physical_object', 'ideal_object', 'geometrical_object', 'material', 'object', 'ordinary_enduring_middle-sized_physical_object', 'particle', 'particular', 'physical_thing', 'scattered_object']
cluster_C = ['context', 'modulus', 'operant_behavior', 'phoneme', 'stimulus', 'stimulation']
cluster_D = ['conceptual_scheme', 'prelinguistic_quality_space']
cluster_E = ['canonical_notation', 'paraphrase', 'concatenation', 'concretion', 'conditional', 'conjunction', 'connective', 'construction', 'contradiction', 'copula', 'form', 'function', 'quantification', 'quantifier', 'quotational', 'predication', 'plural', 'regimentation', 'elimination', 'explication', 'linguistic_form', 'logic', 'syntax', 'variables']

def map_words(words, label, word_label_dict):
    for word in words:
        word_label_dict[word] = label   
    
def get_all_vectors(word_label_dict, model):
    
    vecs = []
    words_in_vocab = []
    
    for word in word_label_dict.keys():
        if word in model.vocab:
            vec = model[word]
            vecs.append(vec)
            words_in_vocab.append(word)
        else:
            print(word, 'oov')
    
    return np.array(vecs), words_in_vocab
    
def main():
    embeddings_path = "../data/n2v_rand_a0.1_n15_s10000_w5_l150_sd1_wd1.txt"
    embeddings = evaluate.load_embeddings(embeddings_path)
    
    word_label_dict = dict()
    map_words(cluster_A, 'language', word_label_dict)
    map_words(cluster_B, 'ontology', word_label_dict)
    map_words(cluster_C, 'reality', word_label_dict)
    map_words(cluster_D, 'mind', word_label_dict)
    map_words(cluster_E, 'metalinguistic', word_label_dict)
    
    for label, words in word_label_dict.items(): 
        input_words.append(label)
        gold_labels.append(words)

    vecs, words_in_vocab = get_all_vectors(word_label_dict, embeddings)
    
    y_pred = KMeans(n_clusters=5, init='random').fit_predict(vecs)
    
    predicted_clusters = defaultdict(list)
    for word, pred_label in zip(words_in_vocab, y_pred):
        predicted_clusters[pred_label].append(word)
        
    for label, words in predicted_clusters.items():
        with open("../data/evaluation_kmeans.txt", 'a') as output_file:
            output_file.write(embeddings_path)
            for label, words in predicted_clusters.items():
                output_file.write(label, words)
                
    
    
if __name__ == "__main__":
    main()