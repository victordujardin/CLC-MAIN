from sentence_transformers import SentenceTransformer
import pymongo

import config


def connect_to_db_collection(mongo_uri, db, collection_name):
    # initialize db connection
    connection = pymongo.MongoClient(mongo_uri)
    collection = connection[db][collection_name]
    return collection


# keys : '_id', 'instruction', 'input' is the description of symptoms , 'output' a diagnosis
def collection_encoder(collection):
    # define transofrmer model (from https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)
    model = SentenceTransformer(config.embedding_model)

    encoded = [x["_id"] for x in collection.find() if "medical_embedding" in x.keys()]
    print(len(encoded))

    for x in collection.find({config.index_fields: {"$exists": False}}, {}):
        # checking if vector already computed for this doc
        if "medical_embedding" not in x.keys():
            if "output" in x.keys():
                med_id = x["_id"]
                diagnosis = x["output"]
                text = diagnosis
                symptoms = None

                # if symptoms field present, concat it with title
                if "input" in x.keys():
                    symptoms = x["input"]
                    text = text + ". " + symptoms

                vector = model.encode(text).tolist()

                collection.update_one(
                    {"_id": med_id},
                    {
                        "$set": {
                            "medical_embedding": vector
                        }
                    },
                    upsert=True,
                )


def compute_encoding(uri, db_n, coll_name):
    collection = connect_to_db_collection(uri, db_n, coll_name)
    collection_encoder(collection)


'''
if __name__ == '__main__':
    mongo_uri = config.mongo_uri
    # db = config.db_name
    db_name = 'tutorial'
    # collection_name = config.collection_name
    collection_name = 'medical_tutorial'
    compute_encoding(mongo_uri, db_name, collection_name)
'''
