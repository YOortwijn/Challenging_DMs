# Challenging Distributional Models with a Conceptual Network of Philosophical Terms
This repository contains the code and evaluation dataset for the paper "Challenging Distributional Models with a Conceptual Network of Philosophical Terms". 

## Corpus
For this paper, we use version 0.5 of the Quine in Context corpus consisting of 228 books and articles by Quine, containing 2,150,356 word tokens and 38,791 word types. It is a high quality corpus where scanned texts were OCR-processed and corrected manually. The corpus was derived from copyrighted works by [Betti et al., 2020](https://www.aclweb.org/anthology/2020.coling-main.586). The corpus is available to researchers that can show they own the original works. Replication instructions are available here: https://github.com/YOortwijn/QuiNE-ground-truth

All code for further preprocessing specific to this paper can be found in the "preprocessing" folder. 

## Conceptual Network 
The philosophical expert categorized words from the index of Quine's Word & Object (1960) as either belonging to one of five clusters (*language*, *ontology*, *reality*, *mind*, *meta-linguistic*) or as a relational term (i.e. part of either the *reference* or *regimentation* relation that connects (parts) of clusters to each other). All data pertaining the conceptual network can be found in the "ground truth" folder. This contains a read me with the motivation behind the categorisation, and the expert-constructed ground truth. 

## Models


## Hyperparameter Tuning


## Evaluation
The tuned models were evaluated against the ground truth with different tasks. The code for the tasks and evaluation can be found in the "evaluation" folder. 

## Contributers 
- Yvette Oortwijn
- Jelke Bloem
- Pia Sommerauer 
- Francois Meyer
- Wei Zhou
- Antske Fokkens

## Contact
yvette [dot] oortwijn [at] gmail [dot] com
