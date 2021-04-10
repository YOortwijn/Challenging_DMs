#!/usr/bin/env python

"""
| File name: grid_search.py
| Author: Francois Meyer
| Email: francoisrmeyer@gmail.com
| Project: The semantics of meaning
| Date created: 30-05-2019
| Date last modified: 30-05-2019
| Python Version: 3.6
"""

import sys
import semantics
from semantics import SemanticSpace
import context
import random
import re
import itertools
import json
from sklearn.preprocessing import normalize
import numpy as np
import time

BACKGROUND_PATH = "../embeddings/wiki_all.model/wiki_all.sent.split.model"
from gensim.models import KeyedVectors

import gensim
import pandas as pd
from semantics import similarity


def load_embeddings(embeddings_path):
    embeddings = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)
    return embeddings


def load_clusters(clusters_path):

    concepts_df = pd.read_csv(clusters_path, header=0)
    concepts_df = concepts_df.dropna(axis=1, how='all')

    concepts = {}
    concepts["A"] = concepts_df["cluster a: language"].dropna().tolist()
    concepts["B"] = concepts_df["cluster b: ontology"].dropna().tolist()
    concepts["C"] = concepts_df["cluster c: reality"].dropna().tolist()
    concepts["D"] = concepts_df["cluster d: mind"].dropna().tolist()
    concepts["E"] = concepts_df["cluster e: on language"].dropna().tolist()

    clusters = list(concepts.keys())

    return clusters, concepts

def load_clusters_extended(clusters_ext_path):

    concepts_df = pd.read_csv(clusters_ext_path, header=0)
    concepts_df = concepts_df.dropna(axis=1, how='all')

    concepts = {}
    concepts["A"] = concepts_df["cluster a: language"].dropna().tolist()
    concepts["B"] = concepts_df["cluster b: ontology"].dropna().tolist()
    concepts["C"] = concepts_df["cluster c: reality"].dropna().tolist()
    concepts["D"] = concepts_df["cluster d: mind"].dropna().tolist()
    concepts["E"] = concepts_df["cluster e: on language"].dropna().tolist()
    
    relations = {}
    relations["reference"] = concepts_df["related: reference"].dropna().tolist()
    
    mixeds = {}
    mixeds["mixed"] = concepts_df["related: regimentation"].dropna().tolist()
    
    clusters = list(concepts.keys())
    related = list(relations.keys())
    mixed = list(mixeds.keys())
    
    return clusters, concepts, related, relations, mixed, mixeds

def evaluate_embeddings(embeddings, clusters, concepts):

    total = 0
    correct = 0

    for cluster in clusters:
        for concept1 in concepts[cluster]:

            #print(concept)

            similar_concepts = list(set(concepts[cluster]) - set([concept1]))
            for similar_concept in similar_concepts:

                different_clusters = list(set(clusters) - set([cluster]))
                different_concepts = []
                for different_cluster in different_clusters:
                    different_concepts.extend(concepts[different_cluster])
                    for different_concept in different_concepts:

                        if not (concept1 in embeddings and similar_concept in embeddings and different_concept in embeddings):
                            continue

                        similar_sim = similarity(embeddings[concept1], embeddings[similar_concept])
                        different_sim = similarity(embeddings[concept1], embeddings[different_concept])

                        print(concept1)
                        print(similar_concept)
                        print(different_concept)
            
                        total += 1
                        if similar_sim > different_sim:
                            correct += 1
                            print("-> correct")
                            print()
            
                        else:
                            print()
                
                
    print(total)
    print(correct)

    acc = correct / total
    return acc


def evaluate_embeddings_extended(embeddings, clusters, concepts, related, relations, mixed, mixeds):

    score_total = 0
    score_clustered = 0
    score_related = 0
    score_none = 0
    
    pair_missing = 0
    
    for cluster in clusters:
        for concept in concepts[cluster]:

            similar_concepts = list(set(concepts[cluster]) - set([concept]))
            for similar_concept in similar_concepts:
                

                different_clusters = list(set(clusters) - set([cluster]))
                for different_cluster in different_clusters:
                    different_concepts = concepts[different_cluster]
                    for different_concept in different_concepts:
                        

                        if not (concept in embeddings and similar_concept in embeddings and different_concept in embeddings):
                            pair_missing += 1
                            continue

                        similar_sim = similarity(embeddings[concept], embeddings[similar_concept])
                        different_sim = similarity(embeddings[concept], embeddings[different_concept])

                        print(concept)
                        print(similar_concept)
                        print(different_concept)
            
                        score_total += 1
                        if similar_sim > different_sim:
                            score_clustered += 1
                            print("-> clustered")
                            print()
                            
                        else:
                            print()    
                            
                            for r in related:     
                                if concept in relations[r] and different_concept in relations[r]:                                                 
                    
                                    score_related += 1
                                    print("-> related")
                                    print()
                                
                                else:
            
                                    for mi in mixed:
                                        if concept in mixeds[mi] and different_concept in mixeds[mi]:
                    
                                            score_related += 1
                                            print("-> related")
                                            print()
            
                        
                
                            
                
    

    per_clustered = score_clustered / score_total
    per_related = score_related / score_total
    #per_pair_missing = pair_missing / score_total
    
    
    
    print(f"total: {score_total}")
    print(f"clustered: {score_clustered}")
    print(per_clustered)
    print(f"related: {score_related}")
    print(per_related)
    print()
    print(f"pair missing: {pair_missing}")
    #print(per_pair_missing)
    
    #return per_clustered
    #return per_related
    #return per_mixed

    
def check_embeddings(clusters, concepts, embeddings):
    for cluster in clusters:
        for concept in concepts[cluster]:
            if not concept in embeddings:
                print(f"no embedding: {concept}")
                
            if concept in embeddings:
                print(concept)                
            

def main():

    #model = sys.argv[1]
    embeddings_path = "../data/n2v_qnew_a0.1_n15_s10000_w5_l150_sd1_wd1_combined.txt"
    embeddings = load_embeddings(embeddings_path)

    #clusters_path = "../data/terms.csv"
    #clusters1, concepts1 = load_clusters(clusters_path)
    
    clusters_ext_path = "../data/terms_ext.csv"
    clusters2, concepts2, related2, relations2, mixed2, mixes2 = load_clusters_extended(clusters_ext_path)

    #acc = evaluate_embeddings(embeddings, clusters1, concepts1)
    #print(acc)
    
    #acc_ext = evaluate_embeddings_extended(embeddings, clusters2, concepts2)
    #print(acc_ext)
    acc_ext = evaluate_embeddings_extended(embeddings, clusters2, concepts2, related2, relations2, mixed2, mixes2)


if __name__ == "__main__":
    main()