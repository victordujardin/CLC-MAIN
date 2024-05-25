import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from langchain_community.llms import Ollama

import encoder
import config


@st.cache_resource
def initialize_mongo():
    # Initialize MongoDB client
    mongo_uri = config.mongo_uri
    client = MongoClient(mongo_uri)
    db_name = config.db_name
    collection_name = config.collection_name

    collection = client[db_name][collection_name]

    # encode database collection for Vector search
    encoder.collection_encoder(collection)

    # initialize text embedding model (encoder)
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    index_name = "vector_index"
    vector_field_name = "med_embedding"
    text_field_name = "input"

    # specify the MongoDB Atlas database and collection for vector search
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=index_name,
        embedding_key=vector_field_name,
        text_key=text_field_name,
    )
    return vector_store


# Streamlit App
def main():
    # call init function - only once
    vector_store = initialize_mongo()
    # callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # run the LLM from Ollama
    llm = Ollama(model=config.llm_model, callback_manager=callback_manager, base_url = 'http://ollama_service:11434')
    

    st.title("Medical Diagnosis GPT App")

    # user input
    query = st.text_input("Enter your query:")

    # retrieve context data from MongoDB Atlas Vector Search
    retriever = vector_store.as_retriever()

    # query LLM with user input and context data
    if st.button("Query LLM"):
        with st.spinner("Querying LLM..."):
            qa = RetrievalQA.from_chain_type(
                llm, chain_type="stuff", retriever=retriever
            )

            response = qa({"query": query})

            st.text("Llm model Response:")
            st.text(response["result"])


if __name__ == "__main__":
    main()
