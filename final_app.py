import time

import chromadb
from chromadb.config import Settings
import ollama
import gradio as gr
import uuid
from collections import defaultdict
import os
import shutil
from logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# Local imports
from semantic_analysis import build_semantic_docs
from xml_to_json import convert_xml_to_json
from datetime import datetime
from zoneinfo import ZoneInfo

# -----------------------------------
# DATA INGESTION LOGIC
# -----------------------------------

def get_data_from_xml(xml_path, json_path):
    """
    Parses XML to JSON and extracts semantic documents.
    """
    logger.info(f"Processing XML file: {xml_path}")
    doc = convert_xml_to_json(xml_path, json_path)
    if doc:
        logger.info("XML conversion successful, building semantic documents")
        docs = build_semantic_docs(json_path)
        logger.info(f"Generated {len(docs) if docs else 0} semantic documents")
        return docs
    logger.error("XML conversion failed")
    return None

def group_documents_by_loinc(semantic_docs):
    """
    Groups semantic documents by LOINC code and aggregates them into sections.
    """
    logger.info(f"Grouping {len(semantic_docs)} documents by LOINC code")
    grouped = defaultdict(list)

    for doc in semantic_docs:
        loinc = doc["metadata"].get("loinc", "UNKNOWN")
        grouped[loinc].append(doc)

    logger.debug(f"Found {len(grouped)} unique LOINC codes")
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

    logger.info(f"Created {len(final_docs)} grouped section documents")
    return final_docs

def create_ephemeral_collection(section_docs):
    """
    Creates a new ephemeral collection and stores documents.
    Returns the collection object.
    """
    logger.info("Creating ephemeral ChromaDB collection")
    try:
        # Ephemeral / In-Memory Client
        if hasattr(chromadb, 'EphemeralClient'):
            client = chromadb.EphemeralClient()
            logger.debug("Using EphemeralClient")
        else:
            client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
            logger.debug("Using standard Client with ephemeral settings")
            
        # Create unique collection name for this session/upload to avoid conflicts
        # or just use a standard one since the client is ephemeral to the object
        collection_name = f"session_{uuid.uuid4().hex}"
        logger.debug(f"Collection name: {collection_name}")
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"type": "CCDA_SECTION_LEVEL"}
        )
        logger.info(f"Created collection: {collection_name}")

        if section_docs:
            logger.info(f"Adding {len(section_docs)} documents to collection")
            collection.add(
                ids=[doc["id"] for doc in section_docs],
                documents=[doc["document"] for doc in section_docs],
                metadatas=[doc["metadata"] for doc in section_docs]
            )
            logger.info("Successfully added all documents to collection")
        
        return collection
        
    except Exception as e:
        logger.error(f"Error creating collection: {e}", exc_info=True)
        print(f"Error creating collection: {e}")
        return None

# -----------------------------------
# UPLOAD HANDLER
# -----------------------------------

def process_upload(file_obj, history):
    """
    Handles the file upload event.
    Returns: (updated_history, collection_state, specific message)
    """
    logger.info("File upload initiated")
    if file_obj is None:
        logger.warning("No file provided in upload")
        return history, None

    # Determine file path (Gradio provides a NamedString or temp path)
    if hasattr(file_obj, 'name'):
        xml_path = file_obj.name
    else:
        xml_path = file_obj # In some versions it's just the path string

    logger.info(f"Processing uploaded file: {os.path.basename(xml_path)}")
    # Define a temp json path
    json_path = xml_path + ".json"

    history = history or []
    history.append({"role": "user", "content": f"Uploaded file: {os.path.basename(xml_path)}"})
    history.append({"role": "assistant", "content": "Processing file... please wait."})

    try:
        # 1. Convert & Extract
        logger.info("Step 1: Converting XML and extracting data")
        docs = get_data_from_xml(xml_path, json_path)
        
        if not docs:
            logger.error("Failed to extract documents from XML")
            history[-1]["content"] = "Error: Failed to process XML file. Make sure it is a valid CCDA."
            return history, None

        # 2. Group
        logger.info("Step 2: Grouping documents by LOINC")
        final_docs = group_documents_by_loinc(docs)
        
        # 3. Store in DB
        logger.info("Step 3: Storing documents in ChromaDB")
        collection = create_ephemeral_collection(final_docs)
        
        if collection:
            doc_count = len(final_docs)
            logger.info(f"Upload successful: {len(docs)} records -> {doc_count} section documents")
            history[-1]["content"] = f"✅ Success! Processed {len(docs)} records into {doc_count} section documents.\nSystem is ready. Ask me anything about the patient."
            return history, collection
        else:
            logger.error("Database initialization failed")
            history[-1]["content"] = "Error: Database initialization failed."
            return history, None
            
    except Exception as e:
        logger.exception(f"Critical error during file processing: {e}")
        history[-1]["content"] = f"Critical Error during processing: {str(e)}"
        return history, None


# -----------------------------------
# QUERY OPTIMIZATION LAYER
# -----------------------------------

def optimize_search_query(user_query):
    """
    Uses LLM to extract key search terms from a complex user query.
    Example: "What is the patient's full name?" -> "patient full name demographics"
    """
    logger.debug(f"Optimizing search query: '{user_query}'")
    system_prompt = """
    You are a Search Query Optimizer for a clinical database.
    Your job is to convert the user's natural language question into a concise list of KEYWORDS for vector search.
    
    Rules:
    1. Remove stop words (the, a, is, of, etc.).
    2. Extract only the core clinical entities and properties.
    3. Add relevant section names if implied (e.g. "age" -> "age demographics", "meds" -> "medication", "shots" -> "immunization").
    4. OUTPUT ONLY THE KEYWORDS. No explanations.
    
    Example:
    User: "What is the patient's dob?"
    Output: patient date of birth demographics
    """
    
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        optimized = response["message"]["content"].strip()
        logger.info(f"Query optimized: '{user_query}' -> '{optimized}'")
        return optimized
    except Exception as e:
        logger.error(f"Query optimization failed: {e}")
        print(f"Query optimization failed: {e}")
        return user_query  # Fallback to original query


def chat_stream(user_message, history, collection_state):
    logger.info(f"Chat query received: '{user_message}'")
    
    if collection_state is None:
        logger.warning("Query attempted without uploaded file")
        history = history or []
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "⚠️ Please upload a CCDA XML file first before asking questions."})
        yield history
        return

    history = history or []
    history.append({"role": "user", "content": user_message})

    # Optimize Query
    print(f"Original Query: {user_message}")
    search_query = optimize_search_query(user_message)
    print(f"Optimized Query: {search_query}")

    # Query ChromaDB (Using session-specific collection)
    logger.info(f"Querying ChromaDB with optimized query: '{search_query}'")
    try:
        result = collection_state.query(
            query_texts=[search_query],
            n_results=8
        )
        logger.debug(f"ChromaDB query returned {len(result.get('documents', [[]])[0])} results")
    except Exception as e:
        logger.error(f"Database query error: {e}", exc_info=True)
        history.append({"role": "assistant", "content": f"Database Query Error: {e}"})
        yield history
        return

    retrieved_docs = []
    if result and result.get("documents"):
        for docs in result["documents"]:
            retrieved_docs.extend(docs)

    logger.debug(f"Retrieved {len(retrieved_docs)} documents for context")
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
     for any date reference day and time of today is  {datetime.now(ZoneInfo("Asia/Kolkata"))}

    Clinical Context:
    {context}
    """

    history.append({"role": "assistant", "content": ""})

    logger.info("Generating LLM response")
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
        
        logger.info(f"LLM response generated ({len(assistant_reply)} chars)")
            
    except Exception as e:
        logger.error(f"LLM error: {e}", exc_info=True)
        history[-1]["content"] = f"Error calling LLM: {str(e)} Is Ollama running?"
        yield history


# -----------------------------------
# GRADIO UI
# -----------------------------------

def launch_app():
    logger.info("Launching Gradio application")
    with gr.Blocks(title="Clinical RAG Chat (Streaming)") as demo:
        gr.Markdown("## 🏥 Clinical Information Assistant (CCDA + RAG)")
        gr.Markdown("Upload a CCDA XML file to begin.")

        # Session State to hold the DB collection
        collection_state = gr.State(None)

        with gr.Row():
            file_input = gr.File(
                label="Upload CCDA XML",
                file_types=[".xml"],
                type="filepath" # Returns path as string
            )

        chatbot = gr.Chatbot(height=600)

        msg = gr.Textbox(
            placeholder="Ask a clinical question...",
            label="User Question"
        )

        clear = gr.Button("Clear Chat")

        # 1. File Upload Event
        file_input.upload(
            process_upload,
            inputs=[file_input, chatbot],
            outputs=[chatbot, collection_state] # Update chat with status, store DB in state
        )

        # 2. Chat Submit Event
        msg.submit(
            chat_stream,
            inputs=[msg, chatbot, collection_state], # Pass state to chat function
            outputs=[chatbot]
        ).then(
            lambda: "", None, msg
        )

        clear.click(lambda: ([], None), None, [chatbot, collection_state])

    logger.info("Gradio app configured, starting server")
    demo.launch()
    logger.info("Gradio server started")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Clinical RAG Application Starting")
    logger.info("=" * 50)
    launch_app()
