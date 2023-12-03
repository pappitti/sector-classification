from sentence_transformers import SentenceTransformer
import json
from embeddings import embed_sectors
import numpy as np
from sklearn import svm

with open('GICS.json') as f:
    gics = json.load(f)

#valid embeddings model name, compatible with SentenceTransformer (check on HuggingFace)
model_name='BAAI/bge-small-en-v1.5' #BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5 or all-MiniLM-L6-v2 embedded already
if model_name not in gics[0]["embedding"]:
    print("No embeddings found for this model")
    print("running embeddings.py for this model...")
    embed_sectors(model_name)
    print("Embedding process finished.")

with open('GICS.json') as f:
    gics = json.load(f)

with open('NACE.json') as f:
    nace = json.load(f)

model = SentenceTransformer(model_name)

#prompt
prompt = input("Enter company description: ")

#Classification
classification_choice=input("Choose classification. Enter 1 for GICS, 2 for NACE: ")

if classification_choice == "1":
    classification = gics
elif classification_choice == "2":
    classification = nace
else:
    print("Invalid input. Defaulting to GICS.")
    classification = gics

classification_embeddings = np.array([sector["embedding"][model_name] for sector in classification])
classification_embeddings= classification_embeddings / np.sqrt((classification_embeddings**2).sum(1, keepdims=True)) # normalize embeddings (should be normalized already through SentenceTransformer)

prompt_embeddings = model.encode(prompt)
prompt_embeddings = prompt_embeddings / np.sqrt((prompt_embeddings**2).sum())

similiarity_choice=input("Choose similarity assessment method. Enter 1 for KNN, 2 for SVM: ")
    # what comes next is basically copied from https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb
if similiarity_choice == "1":
    #KNN
    similarities = classification_embeddings.dot(prompt_embeddings)
    sorted_ix = np.argsort(-similarities)
    print("top 5 results:")
    for k in sorted_ix[:5]:
        print(f"row {k}, similarity {classification[k]['metadata']['level4']}")
elif similiarity_choice == "2":
    #SVM
    # create the "Dataset"
    x = np.concatenate([prompt_embeddings[None,...], classification_embeddings]) # x is new array, with query now as the first row
    y = np.zeros(x.shape[0])
    y[0] = 1 # we have a single positive example, mark it as such

    # train your (Exemplar) SVM
    # docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.05, dual="auto")
    clf.fit(x, y) # train

    similarities = clf.decision_function(x)
    sorted_ix = np.argsort(-similarities)
    print("top 5 results:")
    for k in sorted_ix[1:6]:
        print(f"row {k-1}, similarity {classification[k-1]['metadata']['level4']}")