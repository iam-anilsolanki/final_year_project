import chromadb
from chromadb.config import Settings
import ollama
import gradio as gr
import uuid
from collections import defaultdict
from pprint import pprint
import os

# Local imports
# Ensure these files exist in the same directory
from semantic_analysis import build_semantic_docs
from xml_to_json import convert_xml_to_json

# -----------------------------------
# DATA INGESTION LOGIC
# -----------------------------------

def get_data_from_xml(xml_path, json_path):
    """
    Parses XML to JSON and extracts semantic documents.
    """
    doc = convert_xml_to_json(xml_path, json_path)
    if doc:
        docs = build_semantic_docs(json_path)
        return docs
    return None

def group_documents_by_loinc(semantic_docs):
    """
    Groups semantic documents by LOINC code and aggregates them into sections.
    """
    grouped = defaultdict(list)

    for doc in semantic_docs:
        loinc = doc["metadata"].get("loinc", "UNKNOWN")
        grouped[loinc].append(doc)

    final_docs = []

    for loinc, docs in grouped.items():
        section = docs[0]["metadata"].get("section", "UNKNOWN")

        merged_text = []
        merged_text.append(f"SECTION: {section}")
        merged_text.append(f"LOINC: {loinc}")
        merged_text.append("")

        for d in docs:
            merged_text.append(d["document"])
            merged_text.append("")

        final_docs.append({
            "id": str(uuid.uuid4()),
            "document": "\n".join(merged_text).strip(),
            "metadata": {
                "section": section,
                "loinc": loinc,
                "source": "CCDA",
                "aggregation": "LOINC_GROUPED"
            }
        })

    return final_docs

def store_sections_in_chromadb(
    section_docs,
    collection_name="ccda_sections"
):
    """
    Initializes Ephemeral ChromaDB (Non-persistent) and stores the provided documents.
    """
    # 1. Initialize Client (Ephemeral / In-Memory)
    try:
        # Explicitly use EphemeralClient if available, or Client with no settings
        if hasattr(chromadb, 'EphemeralClient'):
            client = chromadb.EphemeralClient()
        else:
            client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
            
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return None

    # 2. Reset Collection (Start fresh)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass # Collection didn't exist

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"type": "CCDA_SECTION_LEVEL"}
    )

    # 3. Batch Add Documents
    if section_docs:
        collection.add(
            ids=[doc["id"] for doc in section_docs],
            documents=[doc["document"] for doc in section_docs],
            metadatas=[doc["metadata"] for doc in section_docs]
        )

    return collection

def initialize_system(xml_path, json_path):
    """
    Main initialization routine to be called once at startup.
    """
    print("--- System Initialization ---")
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return None

    print(f"Step 1: Converting {xml_path} -> {json_path}")
    docs = get_data_from_xml(xml_path, json_path)
    
    if not docs:
        print("Error: Failed to extract documents from XML.")
        return None

    print(f"Step 2: Extracted {len(docs)} semantic documents.")
    final_docs = group_documents_by_loinc(docs)
    print(f"Step 3: Grouped into {len(final_docs)} section-level documents.")
    
    print("Step 4: Storing in ChromaDB...")
    collection = store_sections_in_chromadb(final_docs)
    print("--- System Ready ---")
    return collection

# -----------------------------------
# GLOBAL STATE
# -----------------------------------

# Initialize ONCE when the script runs
XML_FILE = 'CCDA_23103_20Oct2017_1043418.xml'
JSON_FILE = 'CCDA_Converted.json'

try:
    GLOBAL_COLLECTION = initialize_system(XML_FILE, JSON_FILE)
except Exception as e:
    print(f"Initialization Critical Failure: {e}")
    GLOBAL_COLLECTION = None

# -----------------------------------
# STREAMING CHAT SYSTEM
# -----------------------------------

def chat_stream(user_message, history):
    if GLOBAL_COLLECTION is None:
        history = history or []
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "Error: System failed to initialize. Please check the server logs."})
        yield history
        return

    history = history or []
    history.append({"role": "user", "content": user_message})

    # Query ChromaDB
    try:
        result = GLOBAL_COLLECTION.query(
            query_texts=[user_message],
            n_results=5
        )
    except Exception as e:
        history.append({"role": "assistant", "content": f"Database Query Error: {e}"})
        yield history
        return

    retrieved_docs = []
    if result and result.get("documents"):
        for docs in result["documents"]:
            retrieved_docs.extend(docs)

    context = "\n\n".join(retrieved_docs)

    system_prompt = f"""
    You are a clinical information assistant.

    You must answer questions ONLY using the provided clinical context.

    Rules:
    - Do NOT use external knowledge.
    - Do NOT infer or guess.
    - Do NOT hallucinate.
    - If the answer is not explicitly present in the context, reply exactly:
      "The requested information is not available in the provided medical record."

    Answering style:
    - Be concise and factual.
    - Use clinical terminology.
    - List multiple items clearly.
    - Include dates if present. Always convert raw dates (e.g., "20150622") to "YYYY-MM-DD" format (e.g., "2015-06-22").

    Clinical Context:
    {context}
    """

    history.append({"role": "assistant", "content": ""})

    try:
        stream = ollama.chat(
            model="llama3.2",
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        assistant_reply = ""
        for chunk in stream:
            token = chunk["message"]["content"]
            assistant_reply += token
            # Update the last message (assistant's reply)
            history[-1]["content"] = assistant_reply
            yield history
            
    except Exception as e:
        history[-1]["content"] = f"Error calling LLM: {str(e)} Is Ollama running?"
        yield history


# -----------------------------------
# GRADIO UI
# -----------------------------------

def launch_app():
    with gr.Blocks(title="Clinical RAG Chat (Streaming)") as demo:
        gr.Markdown("## 🏥 Clinical Information Assistant (CCDA + RAG)")

        chatbot = gr.Chatbot(height=800)

        msg = gr.Textbox(
            placeholder="Ask a clinical question...",
            label="User Question"
        )

        clear = gr.Button("Clear Chat")

        # Submit handler: updates chatbot and clears textbox
        msg.submit(
            chat_stream,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "", None, msg  # Clear the textbox immediately after submit starts
        )

        clear.click(lambda: [], None, chatbot)

    demo.launch()

if __name__ == "__main__":
    launch_app()
