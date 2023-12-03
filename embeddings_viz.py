import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn import decomposition

with open('GICS.json') as f:
    gics = json.load(f)

with open('NACE.json') as f:
    nace = json.load(f)

#valid embeddings model name, compatible with SentenceTransformer (check on HuggingFace)
model_name='all-MiniLM-L6-v2' #BAAI/bge-small-en-v1.5, BAAI/bge-large-en-v1.5 or all-MiniLM-L6-v2 embedded already

classifications = input("Choose classification. Enter 1 for GICS, 2 for NACE: ")
if classifications == "1":
    classification = gics
elif classifications == "2":
    classification = nace
else:
    print("Invalid input. Defaulting to GICS.")
    classification = gics

sectors_embeddings = np.array([sector["embedding"][model_name] for sector in classification])
pca=decomposition.PCA(n_components=3)
X = pca.fit_transform(sectors_embeddings)

sector_groups={}
k=0
for sector in classification:
    if sector["metadata"]["level1"] in sector_groups:
        sector_groups[sector["metadata"]["level1"]].append((sector["metadata"]["level4"],X[k]))
    else:
        sector_groups[sector["metadata"]["level1"]] = [(sector["metadata"]["level4"],X[k])]
    k+=1

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection='3d')
cmap = plt.get_cmap("tab20")

for i,cat in enumerate(sector_groups.keys()):
    for j,viz in enumerate(sector_groups[cat]):
        label, coordinates = viz
        x, y, z = coordinates
        color = cmap(i / len(sector_groups))
        
        # Scatter plot
        ax.scatter(x, y, z, color=color, label=cat if j == 0 else "")

        # Annotation - commented out because it's too many points
        # ax.text(x, y, z, label, color=color, fontsize=6)
    
ax.legend(bbox_to_anchor=(1.1, 1))
plt.show()