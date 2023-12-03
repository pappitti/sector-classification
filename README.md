# Sector Embeddings 


## The broader project


This repository was built as part of a project to illustrate real-life use cases of AI, working exclusively with basic building blocks in order to demystify this technology. This explains why we reinvent the wheel in many cases. In reality, the transformer library and the Sentence-Transformers libraries have built-in functions to do semantic search. We have not tried to benchmark our DIY approach against the built-in methods. 


Read about other solutions we have built or tested in context of that project to classify companies by sector: https://www.pitti.io/blogs/ELIAAM/sector-classification  


Important notice: these tools can narrow down the options but should not be trusted to identify the correct answer.  


## Creating your own classification assistant
Python files in the repository require PyTorch (https://pytorch.org/get-started/locally/), HugginFace's transformers (https://github.com/huggingface/transformers), sklearn (https://scikit-learn.org/stable/install.html), sentence_transformers(https://www.sbert.net/docs/installation.html) 


Any model that is compatible with Sentence-Transformers can be used with our scripts: if you do not have the model saved locally, it will be fetched from the relevant HuggingFace endpoint and all level 4 sectors of both NACE and GICS classifications will be embedded (embeddings with 3 models, BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5 and all-MiniLM-L6-v2, are saved already in the GICS.json and NACE.json files). Sentence-Transformers is much simpler for a first dip but it is not necessary. Most models on HuggingFace have a model card describing how to achieve the same thing with transformers only. 


## Files description
embeddings.py contains the function that generates embeddings for all level 4 sub-sectors and saves them in the relevant json files.  


single_company.py contains a script that requires setting the model name as variable, and then takes user input for one company description. The user can subsequently choose which classification they want to use, and which methodology they want to use to assess similarity (KNN or SVM).  
 
entire_table.py implements the same methods as single_company.py except that the script iterates through an entire table and saves the top-3 results according to KNN and SVM in a csv format.   


embeddings_viz.py contains the script to visualize the embeddings after dimension reduction with sklearn. Each point represents a level 4 sub-sector, colors are defined based on level 1.  


## Dataset


The GICS and NACE classifications are saved in json files. Descriptions of level 4 sub-sectors are used for embeddings. We also provide a file called portfolio.json, which contains a list of companies scrapped from asset managers websites with information including name, industry according to the asset manager and activity description.  
