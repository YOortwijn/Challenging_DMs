# Challenging Distributional Models with a Conceptual Network of Philosophical Terms
This repository contains the code and data set for the paper "Challenging Distributional Models with a Conceptual Network of Philosophical Terms". 

## The Corpus
The Quine in Context dataset contains all philosophical texts written by W. V. Quine (228 books and articles from between 1932 and 2008) in plaintext format. Every file in the dataset corresponds to one single article or one section, chapter or essay of a book, resulting in a total of 818 files. The dataset has been produced from pdf files, which have been first OCR-ed, and then manually cleaned by the Quine in Context Project team (Katjoesja Kruiger, Suze van Scheers, Lisa Dondorp, Thijs Ossenkoppele, Maud van Lier, Yvette Oortwijn) and by the student cohorts of Arianna Betti's slow-reading class on Quine's Word and Object in 2015/16 and 2016/17 at the University of Amsterdam (https://quine1960.wordpress.com/about/), supervised by the project team. All data that is irrelevant to the content of the article or section has been removed. This includes information on the institution the article or book was published at, repeating headers with the title of the chapter and metadata. Some of the texts are formula-rich and/or symbol-rich: formulas and symbols were replaced by short codes (XfZ and XsZ), which function as place-holders. For precise instructions for cleaning, one can find the manual that was used here. This document contains all the settings to be used in the OCR software ABBYY Finereader, and all the steps for manual correction after processing by ABBYY. Subsequently, paragraph structure was detected and restored using purpose-built normalization scripts geared towards batches of texts displaying ranges of similar shortcomings, ad-hoc command lines and finally visual inspection - at times involving comparison to the book images - and manual editing by means of a good editor.

The texts were converted to [FoLiA XML] (https://www.semanticscholar.org/paper/FoLiA%3A-A-practical-XML-Format-for-Linguistic-a-and-Gompel-Reynaert/941b849d75f3a899e3b2a04bfeb2297ca6da1f02) using the English language module of [UCTO] (https://languagemachines.github.io/ucto/), which also provided sentence segmentation and tokenization. FoLiA markup can represent text as, and maintain unique identifiers for, sections, paragraphs, sentences and words. FoLiA can also represent the lemma of each word. A dedicated [FoLiA XML module] (https://github.com/proycon/spacy2folia) called on [Spacy] (https://spacy.io/) using the core model for English (Spacy English core model: en_core_web_sm} available from its website to provide lemmatization).

## The Conceptual Network 


## The Models


## The Evaluation


## Contributers 
- Yvette Oortwijn
- Jelke Bloem
- Pia Sommerauer 
- Francois Meyer
- Wei Zhou

## Contact
yvette [dot] oortwijn [at] gmail [dot] com
