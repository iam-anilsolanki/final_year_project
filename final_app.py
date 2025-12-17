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
    Returns: (patient_id, patient_name, docs) or (None, None, None) on error
    """
    logger.info(f"Processing XML file: {xml_path}")
    doc = convert_xml_to_json(xml_path, json_path)
    if doc:
        logger.info("XML conversion successful, building semantic documents")
        patient_id, patient_name, docs = build_semantic_docs(json_path)
        logger.info(f"Generated {len(docs) if docs else 0} semantic documents for patient {patient_id} - {patient_name}")
        return patient_id, patient_name, docs
    logger.error("XML conversion failed")
    return None, None, None

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

def create_persistent_collection(section_docs, patient_id, patient_name):
    """
    Creates a persistent ChromaDB collection for a specific patient.
    Collection name format: patient_<ID> (e.g., patient_23103)
    If collection exists, it will be deleted and recreated (REPLACE strategy).
    Returns the collection object.
    """
    logger.info(f"Creating persistent ChromaDB collection for patient {patient_id} - {patient_name}")
    try:
        # Persistent Client - stores data in ./chroma_db directory
        client = chromadb.PersistentClient(path="./chroma_db")
        logger.debug("Using PersistentClient with path: ./chroma_db")
            
        # Use patient-specific collection name
        collection_name = f"patient_{patient_id}"
        logger.debug(f"Collection name: {collection_name}")
        
        # Delete existing collection if it exists (REPLACE strategy)
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"Deleted existing collection for patient {patient_id} (REPLACE strategy)")
        except Exception:
            logger.debug(f"No existing collection to delete: {collection_name}")
            pass
        
        # Create new collection with patient metadata
        from datetime import datetime
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "type": "CCDA_SECTION_LEVEL",
                "patient_id": patient_id,
                "patient_name": patient_name,
                "last_upload": datetime.now().isoformat()
            }
        )
        logger.info(f"Created persistent collection: {collection_name}")

        if section_docs:
            logger.info(f"Adding {len(section_docs)} documents to collection")
            # Add upload timestamp to each document's metadata
            upload_time = datetime.now().isoformat()
            for doc in section_docs:
                doc["metadata"]["upload_timestamp"] = upload_time
            
            collection.add(
                ids=[doc["id"] for doc in section_docs],
                documents=[doc["document"] for doc in section_docs],
                metadatas=[doc["metadata"] for doc in section_docs]
            )
            logger.info("Successfully added all documents to persistent collection")
        
        return collection
        
    except Exception as e:
        logger.error(f"Error creating persistent collection: {e}", exc_info=True)
        print(f"Error creating collection: {e}")
        return None

def list_patient_collections():
    """
    Lists all existing patient collections in ChromaDB.
    Returns: List of dicts with patient info: [{"id": "23103", "name": "Alice Newman", "display": "23103 - Alice Newman"}, ...]
    """
    logger.info("Listing all patient collections")
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        all_collections = client.list_collections()
        
        patients = []
        for coll in all_collections:
            # Filter only patient collections (pattern: patient_*)
            if coll.name.startswith("patient_"):
                metadata = coll.metadata or {}
                patient_id = metadata.get("patient_id", coll.name.replace("patient_", ""))
                patient_name = metadata.get("patient_name", "Unknown")
                
                patients.append({
                    "id": patient_id,
                    "name": patient_name,
                    "display": f"{patient_id} - {patient_name}",
                    "last_upload": metadata.get("last_upload", "Unknown")
                })
        
        logger.info(f"Found {len(patients)} patient collections")
        return patients
        
    except Exception as e:
        logger.error(f"Error listing patient collections: {e}", exc_info=True)
        return []

def get_collection_by_patient_id(patient_id):
    """
    Retrieves an existing patient collection by patient ID.
    Returns: (collection, patient_name) or (None, None) if not found
    """
    logger.info(f"Loading collection for patient {patient_id}")
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection_name = f"patient_{patient_id}"
        collection = client.get_collection(name=collection_name)
        
        metadata = collection.metadata or {}
        patient_name = metadata.get("patient_name", "Unknown")
        
        logger.info(f"Successfully loaded collection for patient {patient_id} - {patient_name}")
        return collection, patient_name
        
    except Exception as e:
        logger.error(f"Error loading collection for patient {patient_id}: {e}")
        return None, None

# -----------------------------------
# UPLOAD HANDLER
# -----------------------------------

def process_upload(file_obj, history):
    """
    Handles the file upload event.
    Returns: (updated_history, collection_state, patient_info_dict)
    """
    logger.info("File upload initiated")
    if file_obj is None:
        logger.warning("No file provided in upload")
        return history, None, None

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
        # 1. Convert & Extract (including patient info)
        logger.info("Step 1: Converting XML and extracting data")
        patient_id, patient_name, docs = get_data_from_xml(xml_path, json_path)
        
        if not docs or not patient_id:
            logger.error("Failed to extract documents or patient info from XML")
            history[-1]["content"] = "Error: Failed to process XML file. Make sure it is a valid CCDA."
            return history, None, None

        # Check if patient already exists
        existing_collection, _ = get_collection_by_patient_id(patient_id)
        is_update = existing_collection is not None
        
        if is_update:
            logger.info(f"Patient {patient_id} already exists - will update collection")

        # 2. Group
        logger.info("Step 2: Grouping documents by LOINC")
        final_docs = group_documents_by_loinc(docs)
        
        # 3. Store in DB (patient-specific collection)
        logger.info("Step 3: Storing documents in ChromaDB")
        collection = create_persistent_collection(final_docs, patient_id, patient_name)
        
        if collection:
            doc_count = len(final_docs)
            logger.info(f"Upload successful: {len(docs)} records -> {doc_count} section documents")
            
            # Create patient info dict for state
            patient_info = {
                "id": patient_id,
                "name": patient_name,
                "display": f"{patient_id} - {patient_name}"
            }
            
            # Different message for new vs update
            if is_update:
                history[-1]["content"] = f"✅ Updated patient: **{patient_id} - {patient_name}**\n\nProcessed {len(docs)} records into {doc_count} section documents.\n\nSystem is ready. Ask me anything about this patient."
            else:
                history[-1]["content"] = f"✅ New patient added: **{patient_id} - {patient_name}**\n\nProcessed {len(docs)} records into {doc_count} section documents.\n\nSystem is ready. Ask me anything about this patient."
            
            return history, collection, patient_info
        else:
            logger.error("Database initialization failed")
            history[-1]["content"] = "Error: Database initialization failed."
            return history, None, None
            
    except Exception as e:
        logger.exception(f"Critical error during file processing: {e}")
        history[-1]["content"] = f"Critical Error during processing: {str(e)}"
        return history, None, None


def load_existing_patient(patient_selection, history):
    """
    Loads an existing patient's collection when selected from dropdown.
    Returns: (updated_history, collection_state, patient_info_dict)
    """
    logger.info(f"Patient selection: {patient_selection}")
    
    if not patient_selection or patient_selection == "Upload New CCDA":
        logger.info("Upload new CCDA selected")
        return history, None, None
    
    # Extract patient ID from selection (format: "23103 - Alice Newman")
    try:
        patient_id = patient_selection.split(" - ")[0].strip()
        logger.info(f"Loading patient ID: {patient_id}")
        
        collection, patient_name = get_collection_by_patient_id(patient_id)
        
        if collection:
            history = history or []
            history.append({
                "role": "assistant", 
                "content": f"📋 Loaded patient: **{patient_id} - {patient_name}**\n\nYou can now ask questions about this patient's medical records."
            })
            
            patient_info = {
                "id": patient_id,
                "name": patient_name,
                "display": f"{patient_id} - {patient_name}"
            }
            
            logger.info(f"Successfully loaded patient {patient_id}")
            return history, collection, patient_info
        else:
            history = history or []
            history.append({
                "role": "assistant",
                "content": f"❌ Error: Could not load patient {patient_id}. Collection not found."
            })
            return history, None, None
            
    except Exception as e:
        logger.error(f"Error loading patient: {e}", exc_info=True)
        history = history or []
        history.append({
            "role": "assistant",
            "content": f"❌ Error loading patient: {str(e)}"
        })
        return history, None, None


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
    yield history  # Show user message immediately
    
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

    # Show processing indicator immediately
    history.append({"role": "assistant", "content": "🤔 Processing your question..."})
    yield history

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
    
    # Custom CSS for modern, professional UI with enhanced styling
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    body, .gradio-container {
        background-color: #0f172a; /* Slate 900 */
        font-family: 'Inter', sans-serif;
        color: #f1f5f9;
        margin: 0;
        padding: 0;
    }

    /* --- Scoped Styles (Cards & Containers) --- */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
    }
    
    .header-container h1 {
        font-weight: 700;
        color: #38bdf8; /* Sky 400 */
        margin-bottom: 0.5rem;
    }
    
    .card {
        background: rgba(30, 41, 59, 0.7); /* Slate 800 with opacity */
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* --- Chatbot Customization --- */
    .chatbot {
        height: 600px !important;
        background-color: transparent !important;
        border: none !important;
    }

    /* Message Bubbles */
    .message-wrap .message.user {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); /* Sky 500 to 600 */
        color: white;
        border-radius: 12px 12px 0 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .message-wrap .message.bot {
        background: #1e293b; /* Slate 800 */
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
        border-radius: 12px 12px 12px 0;
    }
    
    /* Input Field */
    .input-field textarea {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    .input-field textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }

    /* --- Buttons --- */
    .primary-btn {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    .primary-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4) !important;
    }
    
    .secondary-btn {
        background: #334155 !important; /* Slate 700 */
        border: 1px solid #475569 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    .secondary-btn:hover {
        background: #475569 !important;
    }

    .example-btn {
        background: rgba(56, 189, 248, 0.1) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        color: #38bdf8 !important;
        margin: 4px !important;
        font-size: 0.9em !important;
        border-radius: 20px !important;
        text-align: left !important;
    }
    
    .example-btn:hover {
        background: rgba(56, 189, 248, 0.2) !important;
    }

    /* --- Typography & Utils --- */
    .prose {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    code {
        background: #0f172a !important;
        color: #ec4899 !important; /* Pink 500 */
    }

    /* --- Custom Patient Stats --- */
    .patient-info {
        background: rgba(15, 23, 42, 0.4);
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
    }
    
    .stats-card {
        background: rgba(15, 23, 42, 0.4);
        padding: 12px;
        border-radius: 8px;
        margin-top: 10px;
    }
    
    .stats-card p, .patient-info p {
        margin: 4px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a; 
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155; 
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #475569; 
    }

    /* --- Dropdown Fixes --- */
    .dropdown {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    .dropdown .container {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .dropdown label span {
        color: #94a3b8 !important; /* Label text */
        margin-bottom: 4px;
        display: block;
    }
    
    /* The internal input wrapper */
    .dropdown .wrap-inner {
        background-color: transparent !important;
        border: 1px solid transparent !important;
    }
    
    /* The actual input/display area */
    .dropdown .secondary-wrap {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        color: white !important;
    }

    .dropdown input {
        background-color: transparent !important;
        color: white !important;
    }

    /* SVG Icons (Chevron) */
    .dropdown svg {
        fill: #94a3b8 !important;
    }

    /* The Options Menu */
    .dropdown ul.options {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        margin-top: 4px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5) !important;
        z-index: 9999 !important;
    }
    
    .dropdown li.item {
        background-color: transparent !important;
        color: #e2e8f0 !important;
        padding: 8px 12px !important;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .dropdown li.item:hover {
        background-color: #334155 !important;
        color: #38bdf8 !important;
    }
    
    .dropdown li.item.selected {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 600;
    }
    """
    
    # Helper function to get patient list for dropdown
    def get_patient_choices():
        patients = list_patient_collections()
        choices = ["📤 Upload New CCDA"] + [f"👤 {p['display']}" for p in patients]
        return choices
    
    # Helper function to get patient statistics
    def get_patient_stats(collection):
        if collection is None:
            return "No patient selected"
        
        try:
            count = collection.count()
            metadata = collection.metadata or {}
            
            stats_html = f"""
            <div class='stats-card'>
                <h4 style='margin-top: 0; color: #38bdf8;'>📊 Collection Statistics</h4>
                <p><strong>Documents:</strong> {count}</p>
                <p><strong>Last Upload:</strong> {metadata.get('last_upload', 'Unknown')[:19].replace('T', ' ')}</p>
                <p><strong>Type:</strong> {metadata.get('type', 'Unknown')}</p>
            </div>
            """
            return stats_html
        except Exception as e:
            logger.error(f"Error getting patient stats: {e}")
            return "<div class='stats-card'><p>Error loading statistics</p></div>"
    
    with gr.Blocks(title="Clinical RAG Chat - AI Medical Assistant", theme=gr.themes.Base(), css=custom_css) as demo:
        # Header
        with gr.Row(elem_classes="header-container"):
            gr.Markdown("""
            # 🏥 Clinical Information Assistant
            ### CCDA Medical Records Analysis with AI-Powered RAG
            *Intelligent patient data retrieval and analysis system*
            """)
        
        # Session States
        collection_state = gr.State(None)
        patient_info_state = gr.State(None)
        user_msg_state = gr.State("")  # Store user message for event chain
        
        with gr.Row():
            # Left Sidebar - Patient Selection & Info
            with gr.Column(scale=3, elem_classes="card"):
                gr.Markdown("### 👥 Patient Management")
                
                # Patient Selection Dropdown
                patient_dropdown = gr.Dropdown(
                    label="Select Patient",
                    choices=get_patient_choices(),
                    value="📤 Upload New CCDA",
                    interactive=True,
                    elem_classes="dropdown"
                )
                
                # Refresh button
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh List", size="sm", elem_classes="secondary-btn")
                
                # File Upload
                file_input = gr.File(
                    label="📁 Upload CCDA XML File",
                    file_types=[".xml"],
                    type="filepath",
                    visible=True,
                    elem_classes="file-upload"
                )
                
                gr.Markdown("---")
                
                # Current Patient Display
                current_patient_display = gr.HTML("""
                <div class='patient-info'>
                    <h4 style='margin-top: 0; color: #38bdf8;'>📋 Current Patient</h4>
                    <p style='color: #94a3b8;'>No patient selected</p>
                </div>
                """)
                
                # Patient Statistics
                patient_stats_display = gr.HTML("")
                
                gr.Markdown("---")
                
                # Example Questions
                gr.Markdown("### 💡 Example Questions")
                
                example_questions = [
                    "What is the patient's full name and date of birth?",
                    "List all current medications",
                    "What are the recent vital signs?",
                    "Show all allergies and reactions",
                    "What procedures has the patient undergone?",
                    "List all immunizations"
                ]
                
                example_buttons = []
                for question in example_questions:
                    btn = gr.Button(f"💬 {question}", size="sm", elem_classes="example-btn")
                    example_buttons.append((btn, question))
            
            # Right Side - Chat Interface
            with gr.Column(scale=5):
                with gr.Row(elem_classes="card"):
                    gr.Markdown("### 💬 Chat Interface")
                
                chatbot = gr.Chatbot(
                    height=600,
                    elem_classes="chatbot",
                    show_label=False,
                    avatar_images=("👤", "🤖")  # User and Bot avatars
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="💭 Ask a clinical question about the patient...",
                        label="Your Question",
                        lines=2,
                        max_lines=4,
                        elem_classes="input-field",
                        show_label=False
                    )
                    
                with gr.Row():
                    submit_btn = gr.Button("🚀 Send", elem_classes="primary-btn")
                    clear_btn = gr.Button("🗑️ Clear Chat", elem_classes="secondary-btn")

        # Event Handlers
        
        # 1. Refresh patient list
        def refresh_patients():
            logger.info("Refreshing patient list")
            return gr.Dropdown(choices=get_patient_choices())
        
        refresh_btn.click(
            refresh_patients,
            outputs=[patient_dropdown]
        )
        
        # 2. Patient dropdown selection
        def on_patient_select(selection, history, current_collection, current_patient_info):
            if selection == "📤 Upload New CCDA" or not selection:
                # Reset to upload mode
                return (
                    gr.File(visible=True),
                    """<div class='patient-info'>
                        <h4 style='margin-top: 0; color: #38bdf8;'>📋 Current Patient</h4>
                        <p style='color: #94a3b8;'>No patient selected</p>
                    </div>""",
                    "",
                    history,
                    None,
                    None
                )
            else:
                # Extract patient ID from selection (format: "👤 23103 - Alice Newman")
                patient_id = selection.replace("👤 ", "").split(" - ")[0].strip()
                
                # Load existing patient
                updated_history, collection, patient_info = load_existing_patient(patient_id, history)
                
                if patient_info:
                    patient_html = f"""
                    <div class='patient-info'>
                        <h4 style='margin-top: 0; color: #38bdf8;'>📋 Current Patient</h4>
                        <p><strong>ID:</strong> {patient_info['id']}</p>
                        <p><strong>Name:</strong> {patient_info['name']}</p>
                    </div>
                    """
                    stats_html = get_patient_stats(collection)
                else:
                    patient_html = """<div class='patient-info'>
                        <h4 style='margin-top: 0; color: #ef4444;'>❌ Error</h4>
                        <p>Could not load patient</p>
                    </div>"""
                    stats_html = ""
                
                return (
                    gr.File(visible=False),
                    patient_html,
                    stats_html,
                    updated_history,
                    collection,
                    patient_info
                )
        
        patient_dropdown.change(
            on_patient_select,
            inputs=[patient_dropdown, chatbot, collection_state, patient_info_state],
            outputs=[file_input, current_patient_display, patient_stats_display, chatbot, collection_state, patient_info_state]
        )
        
        # 3. File Upload Event
        def on_file_upload(file_obj, history, dropdown_value):
            updated_history, collection, patient_info = process_upload(file_obj, history)
            
            # Update dropdown to show new/updated patient
            new_choices = get_patient_choices()
            
            if patient_info:
                patient_html = f"""
                <div class='patient-info'>
                    <h4 style='margin-top: 0; color: #10b981;'>✅ Patient Loaded</h4>
                    <p><strong>ID:</strong> {patient_info['id']}</p>
                    <p><strong>Name:</strong> {patient_info['name']}</p>
                </div>
                """
                new_dropdown_value = f"👤 {patient_info['display']}"
                stats_html = get_patient_stats(collection)
            else:
                patient_html = """<div class='patient-info'>
                    <h4 style='margin-top: 0; color: #ef4444;'>❌ Error</h4>
                    <p>Failed to process file</p>
                </div>"""
                new_dropdown_value = "📤 Upload New CCDA"
                stats_html = ""
            
            return (
                updated_history,
                collection,
                patient_info,
                gr.Dropdown(choices=new_choices, value=new_dropdown_value),
                patient_html,
                stats_html
            )
        
        file_input.upload(
            on_file_upload,
            inputs=[file_input, chatbot, patient_dropdown],
            outputs=[chatbot, collection_state, patient_info_state, patient_dropdown, current_patient_display, patient_stats_display]
        )

        # 4. Chat Submit Event
        def submit_message(user_msg, history, collection):
            if not user_msg or not user_msg.strip():
                return user_msg, history, ""
            return "", history, user_msg  # Clear input, keep history, store message in state
        
        msg.submit(
            submit_message,
            inputs=[msg, chatbot, collection_state],
            outputs=[msg, chatbot, user_msg_state]
        ).then(
            chat_stream,
            inputs=[user_msg_state, chatbot, collection_state],
            outputs=[chatbot]
        )
        
        submit_btn.click(
            submit_message,
            inputs=[msg, chatbot, collection_state],
            outputs=[msg, chatbot, user_msg_state]
        ).then(
            chat_stream,
            inputs=[user_msg_state, chatbot, collection_state],
            outputs=[chatbot]
        )

        # 5. Clear Chat
        def clear_chat():
            return [], """<div class='patient-info'>
                <h4 style='margin-top: 0; color: #38bdf8;'>📋 Current Patient</h4>
                <p style='color: #94a3b8;'>No patient selected</p>
            </div>""", "", None, None, gr.Dropdown(value="📤 Upload New CCDA")
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, current_patient_display, patient_stats_display, collection_state, patient_info_state, patient_dropdown]
        )
        
        # 6. Example question buttons - populate the message input
        def set_example(question):
            return question
        
        for btn, question in example_buttons:
            btn.click(
                set_example,
                inputs=gr.State(question),
                outputs=msg
            )

    logger.info("Gradio app configured, starting server")
    demo.launch()
    logger.info("Gradio server started")


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Clinical RAG Application Starting")
    logger.info("=" * 50)
    launch_app()
