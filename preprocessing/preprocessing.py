#!/usr/bin/env python

"""
| File name: preprocessing.py
| Author: Francois Meyer
| Email: francoisrmeyer@gmail.com
| Project: The semantics of meaning
| Date created: 31-01-2018
| Date last modified: 31-01-2018
| Python Version: 3.6
"""

import os
import nltk
import re
import string
import chardet
import sys
import gensim
import json

def extract_sentences_from_file(target_word, file_path, encoding):
    """
    Extract all sentences in a text file that contain the target word (only in its specified form).
    :param target_word: word for which sentences are extracted
    :param file_path: text file from which sentences are extracted
    :param encoding: character encoding scheme used to decode text files
    :return: extracted_sentenes: list of extracted sentences
    """

    # Match strings that contain the target word
    regex = re.compile(r"\b({0})\b".format(target_word), flags=re.IGNORECASE)
    extracted_sentences = []
    incomplete_sentence = None

    for line in open(file_path, "r", encoding=encoding, errors="ignore"):
        sentences = [sentence for sentence in nltk.sent_tokenize(line)]
        if len(sentences) == 0:
            continue

        # Append to sentence started on previous line
        if incomplete_sentence:
            sentences[0] = incomplete_sentence + " " + sentences[0]
            incomplete_sentence = None

        # Check if sentence continues to next line
        if sentences[-1][-1] != ".":
            incomplete_sentence = sentences.pop()

        # Keep sentences that contain the target word
        sentences = [sentence for sentence in sentences if regex.search(sentence)]
        extracted_sentences.extend(sentences)

    return extracted_sentences


def extract_sentences(target_word, corpus_path, output_path, directory=True, corpus_name="", preprocess=True,
                      encoding=None):
    """
    Extract all sentences in a corpus that contain the target word and write them to a text file, one sentence per line.
    :param target_word: word for which sentences are extracted
    :param corpus_path: path of text corpus (either a text file or a directory containing multiple text files)
    :param output_path: path of the text file to which the extracted sentences are written
    :param directory: whether or not the provided corpus path is a directory, as opposed to a single text file
                      if True, use all text files in the provided directory to train the new model
                      if False, use only the provided text file to train the new model
    :param corpus_name: specify the name of the new corpus when a directory is provided as the corpus path (only text
                        files with names starting with this string are considered part of the corpus)
    :param preprocess: whether or not the extracted sentences should be
    :param encoding: character encoding scheme used to decode text files
    :return: total_examples: number of sentences extracted

    """

    # Extract sentences from file/s
    detect_encoding = True if encoding is None else True
    extracted_sentences = []
    if directory:
        for file_name in os.listdir(corpus_path):
            if not file_name.startswith(corpus_name):
                continue

            file_path = os.path.join(corpus_path, file_name)

            # Detect encoding if not specified
            if detect_encoding:
                raw_data = open(file_path, 'rb').read()
                detection = chardet.detect(raw_data)
                encoding = detection['encoding']

            sentences = extract_sentences_from_file(target_word, file_path, encoding)
            extracted_sentences.extend(sentences)
    else:
        extracted_sentences = extract_sentences_from_file(target_word, corpus_path, encoding= None)

    if preprocess:
        extracted_sentences = [preprocess_text(sentence) for sentence in extracted_sentences]

    # Write extracted sentences to file
    with open(output_path, "w") as f:
        for sentences in extracted_sentences:
            f.write("%s\n" % sentences)

    total_examples = len(extracted_sentences)
    return total_examples


def preprocess_text(text):
    """
    Transform text such that it is ready to be used as input by a word embedding model.
    :param text: raw string
    :return: text: preprocessed string
    """
    
    #text = text.lower()  # to lower case
    
    text = re.sub(r"e\.g\.", "EG", text)
    text = re.sub(r"E\.g\.", "EG", text)
    text = re.sub(r"i\.e\.", "IE", text)
    text = re.sub(r"I\.e\.", "IE", text)
    text = re.sub(r"\Wpp\.", " PP", text)
    text = re.sub(r"\Wpp\s\.", " PP", text)
    text = re.sub(r"\WPp\.", " PP", text)
    text = re.sub(r"\WPp\s\.", " PP", text)
    text = re.sub(r"\Wp\.", " P", text)
    text = re.sub(r"\Wp\s\.", " P", text)
    text = re.sub(r"\WP\.", " P", text)
    text = re.sub(r"\WP\s\.", " P", text)
    text = re.sub(r"\Wcf\.", " CF", text)
    text = re.sub(r"\WCf\.", " CF", text)
    text = re.sub(r"\Wcf\s\.", " CF", text)
    text = re.sub(r"\WCf\s\.", " CF", text)
    text = re.sub(r"vol\.", "VOL", text)
    text = re.sub(r"vol\s\.", "VOL", text)
    text = re.sub(r"Vol\.", "VOL", text)
    text = re.sub(r"Vol\s\.", "VOL", text)
    text = re.sub(r"fig\.", "FIG", text)
    text = re.sub(r"fig\s\.", "FIG", text)
    text = re.sub(r"Fig\.", "FIG", text)
    text = re.sub(r"Fig\s\.", "FIG", text)
    text = re.sub(r"etc\.", "ETC", text)
    text = re.sub(r"etc\s\.", "ETC", text)
    text = re.sub(r"Etc\.", "ETC", text)
    text = re.sub(r"Etc\s\.", "ETC", text)
    
    #text = re.sub(r"\d+", " ", text)  # remove digits
    #text = re.sub(r"-", " ", text)  # remove punctuation that split words
    #text = re.sub(r'[^a-zA-Z\s]+', " ", text) # remove all non-alphanumeric and non-space characters
    
    #text = re.sub(r"\s+", " ", text).strip()  # remove excess white spaces
    # text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation (only works for UTF-8 punctuation)
    # Other possibilities: removing stop words, stemming, lemmatisation, POS tagging
    # split cotractions: text = re.sub(r"-", " ", text)
    
    

    return text

def preprocess_line(line, vocab):

    formulatoken = re.compile('\*\s*f\s*\*')
    xmltoken = re.compile('\<[\s/]*[^\<\>]+\s*\>')
    alpha = re.compile('[^a-zA-Z\s]+')
    onechar = re.compile('\s\w\s')
    multispace = re.compile('\s+')

    line = line.lower()
    line = formulatoken.sub(' ', line)
    line = xmltoken.sub(' ', line)
    line = alpha.sub(' ', line)
    line = onechar.sub(' ', line)
    line = multispace.sub(' ', line)

    words = line.split()
    words = [word for word in words if word in vocab]
    line = " ".join(words)

    return line

def get_lines(file_path):
    words = []
    for line in open(file_path, "r", encoding='utf-8', errors="ignore"):
        words.append(line.strip())
    return words

def get_term_forms(file_path):
    with open(file_path) as infile:
        term_dict = json.load(infile)       
        
    my_reverse_dict = dict()
    for k, forms in term_dict.items():
        all_forms = set(forms).union({k})
        full_all_forms = set()
        for w in all_forms:
            full_all_forms.add(w)
            w_upper = w.capitalize()
            full_all_forms.add(w_upper)
            
        
        for form in full_all_forms:
            my_reverse_dict[form] = k
                
    return my_reverse_dict
              

def replace_multi_word(line, form_term_dict):
    forms = form_term_dict.keys()
    length_term_list = []
    for form in forms:
        form_multi = form.split(' ')
        form_length = len(form_multi)
        
        length_term_list.append((form_length, form))
        #print(length_term_list)
        length_term_list_sorted = sorted(length_term_list, reverse=True)
        #print(length_term_list_sorted)
    
    for l, form in length_term_list_sorted:
        term = form_term_dict[form]
        regex = re.compile(r"\b({0})\b".format(form), flags=re.IGNORECASE)
        if regex.search(line):
            line = re.sub(regex, lambda m: m.group(0).replace(form, term), line)
            regex = re.compile(r"\b({0})\b".format(term), flags=re.IGNORECASE)
            line = re.sub(regex, lambda m: m.group(0).replace(" ", "_"), line)         
            
            #print(term)
            #print(line)
            #print()
        #else:
            #print("Did not find term:")
            #print(term)
            #print(line)
    return line.strip()

def sentence_split(line):
    sents_cleaned = line.replace('\n', ' ')
    sentences = nltk.sent_tokenize(sents_cleaned)
    return sentences    

def main():
    background_path = "../embeddings/wiki_all.model/wiki_all.sent.split.model"
    background_space = gensim.models.Word2Vec.load(background_path)
    vocab = background_space.wv.index2word

    terms_path = "../data/quineterms_addforms.json"
    terms = get_term_forms(terms_path)

    #input_dir = sys.argv[1]
    #output_dir = sys.argv[2]
    #input_files = os.listdir(input_dir)

    #for i, input_file_name in enumerate(input_files):
    #    print("%d of %d" % (i, len(input_files)))
    #    input_file_path = os.path.join(input_dir, input_file_name)
    #    output_file_path = os.path.join(output_dir, input_file_name)
    #    open(output_file_path, 'w').close()

        # Write predictions to file
    with open("../data/QuineCorpusSent/quinev05sent_abrev", 'a') as output_file:
        with open("../data/quinev05_input_word.txt") as input_file:
            for line in input_file:
                line = preprocess_text(line)
                line_pp = replace_multi_word(line, terms)
                #if not line_pp.isspace():
               
                sentences = sentence_split(line_pp)
                
                
                for s in sentences:
                    output_file.write("%s\n" % s)

if __name__ == "__main__":
    main()