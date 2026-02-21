"""
Script to embed the knowledge base into ChromaDB.
Run this once to set up the vector store. After that, the app loads from the persisted files.

Usage: python embed_data.py
"""

import pandas as pd
import chromadb
import os

def main():
    # load the csv
    df = pd.read_csv("data/ai_knowledge.csv")
    print(f"Loaded {len(df)} entries from knowledge base")

    # set up chromadb with file persistence
    # using the default embedding function (all-MiniLM-L6-v2) since it works well
    # for this size of dataset and doesn't need an API key
    client = chromadb.PersistentClient(path="./chroma_db")

    # delete if it already exists so we start fresh
    try:
        client.delete_collection("ai_knowledge")
        print("Deleted old collection")
    except:
        pass

    collection = client.get_or_create_collection(
        name="ai_knowledge",
        metadata={"description": "AI and tech knowledge base for the chatbot"}
    )

    # add each entry to the collection
    documents = df["content"].tolist()
    topics = df["topic"].tolist()
    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [{"topic": topic} for topic in topics]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Added {len(documents)} documents to ChromaDB")
    print(f"Collection count: {collection.count()}")
    print("Done! chroma_db/ directory is ready.")

if __name__ == "__main__":
    main()
