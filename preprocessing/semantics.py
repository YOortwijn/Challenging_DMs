#!/usr/bin/env python

"""
| File name: semantics.py
| Author: Francois Meyer
| Email: francoisrmeyer@gmail.com
| Project: The semantics of meaning
| Date created: 14-12-2018
| Date last modified: 31-01-2018
| Python Version: 3.6
"""

import os
import numpy as np

import gensim
from gensim.models.word2vec import LineSentence
from nonce2vec.models import nonce2vec

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

import logging
#logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class SemanticSpace(object):
    """
    Class that contains semantic spaces and the functionality to analyse them. A background space can be loaded,
    different models can be used to train a new spaces, and the movement of a target word embedding can be analysed.
    """

    def __init__(self):
        """
        Initialise new SemanticSpace object by initialising class variables to None (for now).
        """
        self.background_space = None
        self.new_space = None
        self.target_word = None

    def load_background_space(self, path):
        """
        Load and return a pretrained Word2Vec model.
        :param path: path of model
        :return: Word2Vec model
        """
        background_space = gensim.models.Word2Vec.load(path)
        self.background_space = background_space
        return background_space

    def train_new_space(self, background_path, corpus_path, total_examples, target_word, directory=False,
                        corpus_name="", target_only=True, reset_target=True,
                        epochs=1,
                        start_alpha=1.0,
                        end_alpha=1.0,
                        negative=3,
                        sample=1e-3,
                        window=15,
                        lambda_den=70,
                        sample_decay=1.9,
                        window_decay=5):


        """
        Load background space, train it on a new corpus to examine the meaning of a target word in the new corpus.
        :param background_path: path of pretrained model
        :param corpus_path: path of new corpus (text file with one sentence per line or a directory containing multiple
                            such text files) or iterable of sentences
        :param total_examples: number of sentences
        :param target_word: word to be tracked
        :param directory: whether or not the provided corpus path is a directory, as opposed to a single text file
                          if True, use all text files in the provided directory to train the new model
                          if False, use only the provided text file to train the new model
        :param corpus_name: specify the name of the new corpus when a directory is provided as the corpus path (only
                            text files with names starting with this string are considered part of the corpus)
        :param target_only: if True, only train embedding of the target word on the new corpus
                            if False, also continue training all other word embeddings
        :param reset_target: if True, discard the embedding of the target word in the background space and randomly
                             initialise its vector before training on the new corpus
                             if False, continue training the target word vector from its vector in the background space
        :param epochs: number number of iterations of the new corpus
        :param start_alpha: initial learning rate
        :param end_alpha: final learning rate
        :return: newly trained Word2Vec or Nonce2Vec model (depending on training options)
        """

        if isinstance(corpus_path, str):
            # Load corpus
            if directory:
                sentences = Sentences(corpus_path, corpus_name)
            else:
                sentences = LineSentence(corpus_path)
        else:
            sentences = corpus_path

        if target_only: # Nonce2Vec

            # Load model from file
            new_space = nonce2vec.Nonce2Vec.load(background_path)
            new_space.vocabulary = nonce2vec.Nonce2VecVocab.load(new_space.vocabulary)
            new_space.trainables = nonce2vec.Nonce2VecTrainables.load(new_space.trainables)

            # Set hyperparameters for high risk learning
            new_space.negative = negative
            new_space.sample = sample
            new_space.window = window
            new_space.lambda_den = lambda_den
            new_space.sample_decay = sample_decay
            new_space.window_decay = window_decay
            
            #new_space.replication = True
            #new_space.sum_over_set = False 
            #new_space.weighted = False 
            #new_space.beta = False
            

            # precompute negative labels optimization for pure-python training
            new_space.neg_labels = []
            if new_space.negative > 0:
                new_space.neg_labels = np.zeros(new_space.negative + 1)
                new_space.neg_labels[0] = 1.

            # Add nonce
            new_space.vocabulary.nonce = target_word
            new_space.build_vocab(sentences, update=True)

            if not reset_target:
                # We can set the new vector for the target word to its old vector, as we would like to continue training
                # for it from where it was in the background space
                target_index = new_space.wv.vocab[target_word + "_true"].index
                new_space.wv.vectors[-1] = np.copy(new_space.wv.vectors[target_index])
                new_space.wv.vectors_norm = None
                new_space.wv.init_sims()

        else: # Word2vec

            new_space = gensim.models.Word2Vec.load(background_path)

            new_space.negative = negative
            new_space.sample = sample
            new_space.window = window
            new_space.min_count = 5

            new_space.build_vocab(sentences, update=True)

            if reset_target:
                # Replace trained target word vector with randomly generated vector
                np.random.seed(1)
                new_space.wv.vectors[new_space.wv.vocab[target_word].index] = (np.random.rand(
                    new_space.wv.vector_size) - 0.5) / new_space.wv.vector_size
                new_space.wv.vectors_norm = None
                new_space.wv.init_sims()

        new_space.train(sentences, total_examples=total_examples,
                        epochs=epochs,
                        start_alpha=start_alpha,
                        end_alpha=end_alpha)

        self.target_word = target_word
        self.new_space = new_space
        return new_space

    def most_similar(self, space="new", target_word=None, topn=10):
        """
        Find the topn most similar words to the target word in the background or new semantic space.
        :param space: either 'background' or 'new'
        :param target_word: word for which similar words are found
        :param topn: how many similar words are found
        :return: list of tuples containing the similar words and the cosine similarities to the target word
        """
        if space != "new" and space != "background":
            raise ValueError("The parameter space must be either 'background' or 'new'.")

        target_word = self.target_word if target_word is None else target_word

        if space == "new":
            return self.new_space.wv.most_similar(target_word, topn=topn)
        else:
            return self.background_space.wv.most_similar(target_word, topn=topn)

    def target_similarity(self, target_word=None):
        """
        Compute the cosine similarity between the target word's embedding in the background semantic space and in the
        new semantic space.
        :param target_word: word that is compared between the two semantic spaces
        :return: cosine similarity of background and new target word embeddings
        """
        target_word = self.target_word if target_word is None else target_word
        return similarity(self.background_space.wv[target_word], self.new_space.wv[target_word])

    def target_distance(self, target_word=None):
        """
        Compute the cosine distance between the target word's embedding in the background semantic space and in the new
        semantic space.
        :param target_word: word that is compared between the two semantic spaces
        :return: cosine distance of background and new target word embeddings
        """
        target_word = self.target_word if target_word is None else target_word
        return distance(self.background_space.wv[target_word], self.new_space.wv[target_word])


    def get_target_vector(self):
        return self.new_space.wv[self.target_word]

    def word_similarity(self, word1, word2):
        return similarity(self.new_space.wv[word1], self.new_space.wv[word2])


class Sentences(object):
    # A memory-friendly iterator (https://rare-technologies.com/word2vec-tutorial/)

    def __init__(self, corpus_path, corpus_name):
        self.corpus_path = corpus_path
        self.corpus_name = corpus_name

    def __iter__(self):
        """
        Iterate through all lines in the corpus files.
        :return: generator object
        """
        for file_name in os.listdir(self.corpus_path):
            if file_name.startswith(self.corpus_name):
                for line in open(os.path.join(self.corpus_path, file_name)):
                    yield line.split()


def similarity(vector1, vector2):
    # Compute and return cosine similarity between two vectors.
    return float(cosine_similarity([vector1], [vector2])[0, 0])


def distance(vector1, vector2):
    # Compute and return cosine distance between two vectors.
    return 1.0 - float(cosine_similarity([vector1], [vector2])[0, 0])




