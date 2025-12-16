import json
import chromadb
import uuid
import ollama

# 1. SETUP: Load the JSON data
def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


# 2. CHUNKING: Prepare documents for RAG
def prepare_chunks(data):
    chunks = []

    # Extract Global Context (Patient Name & ID) to attach to every chunk
    # We do this so the LLM knows WHO the data belongs to, even if it retrieves just one isolated chunk.
    try:
        patient_role = data['ClinicalDocument']['recordTarget']['patientRole']

        # Safe extraction of name (handles list or dict structures)
        p_name_obj = patient_role['patient']['name']
        first_name = p_name_obj['given'][0] if isinstance(p_name_obj['given'], list) else p_name_obj['given']
        last_name = p_name_obj['family']['#text'] if isinstance(p_name_obj['family'], dict) else p_name_obj['family']

        patient_name = f"{first_name} {last_name}"
        patient_id = patient_role['id']['@extension']
    except KeyError:
        patient_name = "Unknown Patient"
        patient_id = "Unknown ID"

    # Navigate to the clinical sections
    # Path: ClinicalDocument -> component -> structuredBody -> component (List of sections)
    try:
        components = data['ClinicalDocument']['component']['structuredBody']['component']
    except KeyError:
        print("Error: Could not find clinical sections in JSON.")
        return []

    # Ensure components is a list (xml parser might make it a dict if only 1 exists)
    if not isinstance(components, list):
        components = [components]

    for comp in components:
        section = comp.get('section')
        if not section:
            continue

        # --- A. Metadata Extraction ---
        # Get the section title (e.g., "ALLERGIES", "VITAL SIGNS")
        title = section.get('title', 'Unknown Section')
        if isinstance(title, dict): title = title.get('#text', 'Unknown')

        # Get the template ID (useful for strict filtering, e.g., finding only Allergies)
        # We take the first root if available
        template_id = "Unknown"
        t_ids = section.get('templateId')
        if isinstance(t_ids, list) and len(t_ids) > 0:
            template_id = t_ids[0].get('@root')
        elif isinstance(t_ids, dict):
            template_id = t_ids.get('@root')

        # --- B. Content Flattening ---
        # We dump the specific section's JSON to a string.
        # In a production app, you might use an LLM to summarize this into natural language first.
        section_text = json.dumps(section, indent=2)

        # Create the text blob for the Vector Database
        # We prepend the Patient Context so the semantic search matches "Alice's Allergies"
        page_content = f"PATIENT: {patient_name} (ID: {patient_id})\nSECTION: {title}\nDATA:\n{section_text}"

        chunks.append({
            "id": str(uuid.uuid4()),  # Unique ID for Chroma
            "text": page_content,
            "metadata": {
                "source": "CCDA_XML",
                "patient_id": patient_id,
                "patient_name": patient_name,
                "section_title": title,
                "template_root": template_id
            }
        })

    return chunks


# 3. STORAGE & RETRIEVAL: ChromaDB Logic
def main():
    # Load JSON
    json_file = "CCDA_Converted.json"  # Ensure this file exists from previous step
    try:
        data = load_data(json_file)
    except FileNotFoundError:
        print(f"File {json_file} not found. Run the XML conversion script first.")
        return

    # Prepare chunks
    chunks = prepare_chunks(data)
    print(f"Generated {len(chunks)} chunks from clinical document.\n")

    if not chunks:
        return

    # Initialize ChromaDB (Persistent means it saves to disk)
    #
    client = chromadb.PersistentClient(path="./chroma_db")

    # Create or Get a collection
    # We use the default embedding function (all-MiniLM-L6-v2) which works well for general text.
    collection = client.get_or_create_collection(name="patient_records")

    # Add documents to Chroma
    print("Adding documents to ChromaDB...")
    collection.add(
        documents=[chunk['text'] for chunk in chunks],
        metadatas=[chunk['metadata'] for chunk in chunks],
        ids=[chunk['id'] for chunk in chunks]
    )
    print("Success! Data indexed.\n")

    # --- DEMO QUERIES ---

    # Query 1: Natural Language Semantic Search
    query_text = "Which allergies patient have?"
    print(f"--- Query: '{query_text}' ---")

    results = collection.query(
        query_texts=[query_text],
        n_results=1  # Return top 1 match
    )

    section_name = results['metadatas'][0][0]['section_title']
    snippet_ = results['documents'][0][0][:]

    # 2. Optimized System Prompt
    # KEY CHANGE: We explicitly tell the model it is NOT a coder and strictly forbid technical explanations.
    system_prompt = """
    You are a Clinical Data Analyst. Your job is to extract medical facts from structured healthcare data (C-CDA/JSON).

    CRITICAL RULES:
    1. **Answer the medical question directly.** (e.g., "Yes, Penicillin" or "No known allergies").
    2. **DO NOT** explain the data structure, JSON format, or field names (like 'entryRelationship' or 'typeCode').
    3. **DO NOT** write code, Python scripts, or GraphQL queries.
    4. **DO NOT** mention "missing SUBJ typeCode". If the data is empty or purely structural without clinical text, say: "No relevant clinical information found in this record."
    5. Only use the provided context.
    """

    # 3. Optimized User Message
    # KEY CHANGE: We label the snippet clearly so the model knows it is 'Reference Material', not code to be analyzed.
    user_content = f"""
    ### REFERENCE DATA (Clinical Record Snippet)
    Section: {section_name}
    Content:
    {snippet_}

    ---

    ### USER QUESTION
    Based strictly on the medical facts in the reference data above: {query_text}
    """

    # 4. Run the Chat
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]
    )

    print(f"--- Query: '{query_text}' ---")
    print(response['message']['content'])


if __name__ == "__main__":
    main()