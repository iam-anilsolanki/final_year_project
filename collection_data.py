import chromadb

client = chromadb.PersistentClient(path='./chroma_db')
collections = client.list_collections()
for collection in collections:
    print(collection)