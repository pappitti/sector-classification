from sentence_transformers import SentenceTransformer
import json

def embed_sectors(model_name='all-MiniLM-L6-v2'):
    with open('GICS.json') as f:
        gics = json.load(f)

    with open('NACE.json') as f:
        nace = json.load(f)

    model = SentenceTransformer(model_name)

    #Our sentences we like to encode
    gics_sentences = [sector["description"] for sector in gics]
    nace_sentences = [sector["description"] for sector in nace]

    #Sentences are encoded by calling model.encode()
    gics_embeddings = model.encode(gics_sentences)
    nace_embeddings = model.encode(nace_sentences)

    #Saves the embeddings in the sector json files
    for i in range(len(gics_embeddings)):
        # checks if the embedding key already exists for this model
        try:
            gics[i]["embedding"][model_name] = gics_embeddings[i].tolist()
        except KeyError:
            gics[i]["embedding"] = {}
            gics[i]["embedding"][model_name] = gics_embeddings[i].tolist()

    for j in range(len(nace_embeddings)):
        # checks if the embedding key already exists for this model
        try:
            nace[j]["embedding"][model_name] = nace_embeddings[j].tolist()
        except KeyError:
            nace[j]["embedding"] = {}
            nace[j]["embedding"][model_name] = nace_embeddings[j].tolist()

    with open('GICS.json', 'w') as f:
        json.dump(gics, f, indent=4)

    with open('NACE.json', 'w') as f:
        json.dump(nace, f, indent=4)
 
#running the file embeds the sectors for the model specified below
embed_sectors("BAAI/bge-small-en-v1.5")