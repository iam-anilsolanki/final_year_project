import chromadb
from pprint import pprint
import ollama
import json
from pprint import pprint

client = chromadb.PersistentClient(path="./chroma_db")

# collection = client.get_or_create_collection('patient_mpid_324')
# collection.add(ids=['123','234'],
#                documents=['Anil is the intern at Aliteprojects','Eshan is intern at Nabhik Solutions'])
#
# result = collection.query(query_texts=['Who is intern at Aliteprojects'],
#                           n_results=4,
#                           where_document={'$contains':'intern'})
# pprint(result)

# collection = client.get_collection('patient_mpid_324')
# questions = 'Who work at aliteprojects'
# result = collection.query(query_texts=questions,n_results=2)
# result = result['documents'][0][0]
#

with open('section_documents.json', 'rb') as f:
    docs = json.load(f)
collection = client.get_or_create_collection(docs[0]['metadata']['patient_id'])
for doc in docs:
    collection.add(
        ids=[doc["id"]],
        documents=[doc["document"]],
        metadatas=[doc["metadata"]]
    )
response1 = collection.query(query_texts='allergies name',n_results=1)
pprint(response1)

question = 'allergies name'
system_prompt = """
You are a medical RAG assistant reading C-CDA clinical data.

You are given:
- Retrieved context from a vector database
- A user question

Instructions:
- Answer ONLY from the provided context.
- Do not infer, guess, or add explanations.
- Do not rephrase medical terms.
- If multiple values exist, list them.
- If no answer exists, say:
  "No relevant information found in this record."

Output format:
- Direct answer only
- No preamble
- No commentary
"""

response = ollama.chat(
    model='llama3.2',
    messages=[{'role':'system','content':system_prompt},{'role':'user','content':f'context :{response1} and the question: {question}'}]
)
pprint(response['message']['content'])