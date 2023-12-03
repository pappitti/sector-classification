from sentence_transformers import SentenceTransformer
import json, csv, time
from embeddings import embed_sectors
import numpy as np
from sklearn import svm

with open('GICS.json') as f:
    gics = json.load(f)

analyzed_portfolios = {}

#valid embeddings model name, compatible with SentenceTransformer (check on HuggingFace)
model_name='all-MiniLM-L6-v2' #BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5 or all-MiniLM-L6-v2 embedded already
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

#your table
with open('portfolio.json') as f: #update for your file
    portfolio = json.load(f)

#Classification
classification_choice=input("Choose classification. Enter 1 for GICS, 2 for NACE: ")

if classification_choice == "1":
    classification = gics
elif classification_choice == "2":
    classification = nace
else:
    print("Invalid input. Defaulting to GICS.")
    classification = gics

#to measure duration
start_time = time.time()

classification_embeddings = np.array([sector["embedding"][model_name] for sector in classification])
classification_embeddings= classification_embeddings / np.sqrt((classification_embeddings**2).sum(1, keepdims=True)) # normalize embeddings (should be normalized already through SentenceTransformer)

for funds in portfolio: #this must be adapted to the shape of your dataset (see portfolio.json)
    for company in portfolio[funds]: #this must be adapted to the shape of your dataset
        prompt = company["description"]
        analyzed_portfolios[company["company_name"]] = {"description": prompt, "sponsor": funds,"sector":company["sector"]}
    
        prompt_embeddings = model.encode(prompt)
        prompt_embeddings = prompt_embeddings / np.sqrt((prompt_embeddings**2).sum())

        # what comes next is basically copied from https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb
        #this time we run both methodologies
            #KNN
        similarities = classification_embeddings.dot(prompt_embeddings)
        sorted_ix = np.argsort(-similarities)
        analyzed_portfolios[company["company_name"]]["KNN"] = [classification[sorted_ix[0]]['metadata']['level4'],classification[sorted_ix[1]]['metadata']['level4'],classification[sorted_ix[2]]['metadata']['level4']]
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
        analyzed_portfolios[company["company_name"]]["SVM"] = [classification[sorted_ix[1]-1]['metadata']['level4'],classification[sorted_ix[2]-1]['metadata']['level4'],classification[sorted_ix[3]-1]['metadata']['level4']]

end_time = time.time()
print("Duration: ", end_time - start_time)

#writes everything to a csv file
with open("output.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["company","sponsor","description","sector",f"KNN_1_{model_name}",f"KNN_2_{model_name}",f"KNN_3_{model_name}",f"SVM_1{model_name}",f"SVM_2_{model_name}",f"SVM_3_{model_name}"])

    for company in analyzed_portfolios:
        row=[company,analyzed_portfolios[company]["sponsor"],analyzed_portfolios[company]["description"],analyzed_portfolios[company]["sector"],analyzed_portfolios[company]["KNN"][0],analyzed_portfolios[company]["KNN"][1],analyzed_portfolios[company]["KNN"][2],analyzed_portfolios[company]["SVM"][0],analyzed_portfolios[company]["SVM"][1],analyzed_portfolios[company]["SVM"][2]]
        writer.writerow(row)