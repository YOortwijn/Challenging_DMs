#!/usr/bin/env python

"""
| File name: context.py
| Author: Francois Meyer
| Email: francoisrmeyer@gmail.com
| Project: The semantics of meaning
| Date created: 07-03-2018
| Date last modified: 07-03-2018
| Python Version: 3.6
"""

import preprocessing
import numpy as np
from semantics import SemanticSpace
from nltk.corpus import wordnet

import os
import re

"""def get_lines(target_word, corpus_path):
    
    #:param target_word:
    #:param corpus_path:
    #:return: list of lines (strings) containing the target word
    
    multiword = False
    if re.search(r"\s", target_word):
        multiword = True

    lines = []
    regex = re.compile(r"\b({0})\b".format(target_word), flags=re.IGNORECASE)
    for file_name in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, file_name)
        for line in open(file_path, "r", encoding='utf-8', errors="ignore"): 
            if regex.search(line):
                if multiword:
                    line = re.sub(regex, lambda m: m.group(0).replace(" ", "_"), line)
                lines.append(line.strip())
    return lines """

def get_lines(target_word, corpus_path):
    lines = []
    regex = re.compile(r"\b({0})\b".format(target_word), flags=re.IGNORECASE)
    
    for file_name in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, file_name)
        for line in open(file_path, "r", encoding='utf-8', errors="ignore"): 
            if regex.search(line):
                lines.append(line.strip())
    return lines 
                
def get_contexts(target_word, corpus_path, corpus_name="", contexts_path="contexts.txt"):
    total_examples = preprocessing.extract_sentences(target_word, corpus_path, contexts_path, directory=True,
                                                     corpus_name=corpus_name, preprocess=True)

    with open(contexts_path) as contexts_file:
        contexts = contexts_file.readlines()
    contexts = [x.strip() for x in contexts]

    return contexts


def get_frequencies(target_word, contexts):
    """
    return:
    freqs: list of the frequencies of all context words
    mean_freqs: list of mean frequencies of all contexts
    """

    # Use the vocab counts stored in the Word2Vec model
    # NOTE: these are the counts after downsampling has been applid. To get the raw counts, keep_raw_vocab
    # must be set to True in build_vocab and can then be accessed in Word2Vec.vocabulary.raw_vocab.

    background_path = "../embeddings/wiki_all.model/wiki_all.sent.split.model"
    ss = SemanticSpace()
    ss.load_background_space(background_path)

    freqs = []
    mean_freqs = []

    for context in contexts:
        context_words = context.split()
        context_words = [word for word in context_words if word != target_word]

        for word in context_words:
            if word in ss.background_space.wv.vocab:
                freq = ss.background_space.wv.vocab[word].count
                freqs.append(freq)
            else:
                freqs.append(0)

        mean_freq = np.mean(freqs[-len(context_words):])
        mean_freqs.append(mean_freq)

    return freqs, mean_freqs


def get_polysemy_wup(target_word, contexts):
    """
    return:
    synset_counts: list of the synset counts of all context words
    mean_synset_counts: list of mean synset counts of all contexts
    """
    polysemy_scores = []
    mean_polysemy_scores = []

    for context in contexts:
        context_words = context.split()
        context_words = [word for word in context_words if word != target_word]

        for word in context_words:
            synsets = wordnet.synsets(word)
            word_polysemy_scores = []
            for i in range(len(synsets)):
                inverse_wup_scores = []
                for j in range(len(synsets)):
                    if i == j:
                        continue
                    wup_score = synsets[i].wup_similarity(synsets[j])
                    if wup_score is not None:
                        inverse_wup_scores.append(1.0 - wup_score)

                if len(inverse_wup_scores) > 0:
                    synset_score = np.mean(inverse_wup_scores)
                    word_polysemy_scores.append(synset_score)

            polysemy_score = np.sum(word_polysemy_scores)
            polysemy_scores.append(polysemy_score)

        mean_polysemy_score = np.mean(polysemy_scores[-len(context_words):])
        mean_polysemy_scores.append(mean_polysemy_score)

    return polysemy_scores, mean_polysemy_scores


def get_polysemy(target_word, contexts):
    """
    return:
    synset_counts: list of the synset counts of all context words
    mean_synset_counts: list of mean synset counts of all contexts
    """
    synset_counts = []
    mean_synset_counts = []

    for context in contexts:
        context_words = context.split()

        for word in context_words:
            if word == target_word:
                continue
            synset_count = len(wordnet.synsets(word))
            synset_counts.append(synset_count)

        mean_synset_count = np.mean(synset_counts[-len(context_words):])
        mean_synset_counts.append(mean_synset_count)

    return synset_counts, mean_synset_counts


def get_entropy(target_word, contexts):
    """
    return:
    entropy: overall entropy
    context_entropies: list of context entropies
    """

    background_path = "../embeddings/wiki_all.model/wiki_all.sent.split.model"
    ss = SemanticSpace()
    ss.load_background_space(background_path)

    total = 0
    for word in ss.background_space.wv.vocab:
        total += ss.background_space.wv.vocab[word].count

    context_entropies = []

    for context in contexts:
        context_words = context.split()
        context_words = [word for word in context_words if word != target_word]
        context_entropy = 0

        for word in context_words:
            if word in ss.background_space.wv.vocab:
                freq = ss.background_space.wv.vocab[word].count
                p = freq / total
                e = - p * np.log(p)
                e = e / np.log(len(context_words)+1)
                context_entropy += e

        context_entropies.append(context_entropy)

    entropy = sum(context_entropies)
    return entropy, context_entropies


def get_context_lengths(target_word, contexts):
    """
    :param contexts: list of contexts of a target word
    :return: list of context lengths
    """
    context_lengths = []

    for context in contexts:
        context_words = context.split()
        context_words = [word for word in context_words if word != target_word]
        context_length = len(context_words)
        context_lengths.append(context_length)

    return context_lengths


def get_unique_counts(target_word, contexts):

    unique_counts = []

    for context in contexts:
        context_words = context.split()
        context_words = [word for word in context_words if word != target_word]
        context_unique_count = len(set(context_words))
        unique_counts.append(context_unique_count)

    return unique_counts


def main():
    print()
    target_word = "ik zukkel zo"
    line = "ik zukkel zo en ek zukkel niet vaak daar ben zo waar een grote problemen Ik zukkel zo vandaag, tog zukkel ik zo, ja ik zukkel zo!"
    pat = re.compile(r"\b({0})\b".format(target_word), flags=re.IGNORECASE)
    print(re.sub(pat, lambda m: m.group(0).replace(" ", "_"), line))

if __name__ == '__main__':
    main()

