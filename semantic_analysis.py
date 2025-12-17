import json
import uuid
from datetime import datetime
from logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

def format_date(date_str):
    """
    Parses a date string (YYYYMMDD) and returns it in YYYY-MM-DD format.
    If parsing fails, returns the original string.
    """
    if not date_str:
        return "Unknown Date"
    try:
        # Attempt to parse YYYYMMDD
        if len(date_str) >= 8:
             # Basic handling for 20150622 -> 2015-06-22
             # Also handles timestamps like 201506221000 by slicing first 8 chars
             clean_date = date_str[:8]
             dt = datetime.strptime(clean_date, "%Y%m%d")
             formatted = dt.strftime("%Y-%m-%d")
             logger.debug(f"Formatted date: {date_str} -> {formatted}")
             return formatted
        return date_str
    except ValueError as e:
        logger.warning(f"Failed to format date '{date_str}': {e}")
        return date_str

def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def get_sections(ccda):
    return as_list(
        ccda["ClinicalDocument"]["component"]
        ["structuredBody"]["component"]
    )

def make_doc(text, section, loinc):
    return {
        "id": str(uuid.uuid4()),
        "document": text.strip(),
        "metadata": {
            "section": section,
            "loinc": loinc,
            "source": "CCDA"
        }
    }

# ---------------- ALLERGIES ----------------
def semantic_allergies(section):
    docs = []
    for entry in as_list(section.get("entry")):
        obs = entry["act"]["entryRelationship"]["observation"]
        substance = obs["participant"]["participantRole"]["playingEntity"]["code"]["@displayName"]
        reaction = obs["entryRelationship"][1]["observation"]["value"]["@displayName"]
        severity = obs["entryRelationship"][0]["observation"]["value"]["@displayName"]

        text = f"""
Allergy Record
Substance: {substance}
Reaction: {reaction}
Severity: {severity}
"""
        docs.append(make_doc(text, "ALLERGIES", "48765-2"))
    return docs

# ---------------- ENCOUNTERS ----------------
def semantic_encounters(section):
    docs = []
    for entry in as_list(section.get("entry")):
        enc = entry["encounter"]
        diagnosis = enc["entryRelationship"]["act"]["entryRelationship"]["observation"]["value"]["@displayName"]
        date = format_date(enc["effectiveTime"]["@value"])

        text = f"""
Encounter
Diagnosis: {diagnosis}
Date: {date}
"""
        docs.append(make_doc(text, "ENCOUNTERS", "46240-8"))
    return docs

# ---------------- FUNCTIONAL STATUS ----------------
def semantic_functional_status(section):
    docs = []
    org = section["entry"]["organizer"]
    for comp in as_list(org.get("component")):
        obs = comp["observation"]
        status = obs["value"]["@displayName"]
        date = format_date(obs["effectiveTime"]["@value"])

        text = f"""
Functional Status
Finding: {status}
Recorded On: {date}
"""
        docs.append(make_doc(text, "FUNCTIONAL_STATUS", "47420-5"))
    return docs

# ---------------- IMMUNIZATIONS ----------------
def semantic_immunizations(section):
    docs = []
    for entry in as_list(section.get("entry")):
        sa = entry["substanceAdministration"]
        vaccine = sa["text"]
        date = format_date(sa["effectiveTime"]["@value"])

        text = f"""
Immunization
Vaccine: {vaccine}
Date: {date}
"""
        docs.append(make_doc(text, "IMMUNIZATIONS", "11369-6"))
    return docs

# ---------------- MEDICAL EQUIPMENT ----------------
def semantic_equipment(section):
    docs = []
    # Sometimes this path differs, keeping original logic but wrapped in try just in case? 
    # Original logic seemed bespoke. Assuming it works.
    try:
        device = section["entry"]["organizer"]["component"]["supply"]["participant"]["participantRole"]["playingDevice"]["code"]["@displayName"]
        text = f"""
Medical Equipment
Device: {device}
"""
        docs.append(make_doc(text, "MEDICAL_EQUIPMENT", "46264-8"))
    except KeyError:
        pass # Handle cases where structure varies slightly
    return docs

# ---------------- MEDICATIONS ----------------
def semantic_medications(section):
    docs = []
    for entry in as_list(section.get("entry")):
        sa = entry["substanceAdministration"]
        
        # Safely extract med name
        try:
             med = sa["consumable"]["manufacturedProduct"]["manufacturedMaterial"]["code"]["@displayName"]
        except KeyError:
             med = "Unknown Medication"

        route = sa.get("routeCode", {}).get("@displayName", "Unknown Route")
        
        # Original code assumed list for effectiveTime: sa["effectiveTime"][0]["low"]["@value"]
        # We'll make it robust
        start = "Unknown Date"
        try:
            eff_time = as_list(sa.get("effectiveTime"))
            if eff_time and isinstance(eff_time[0], dict):
                 start = eff_time[0].get("low", {}).get("@value", "Unknown")
            elif eff_time:
                 # Fallback if structure is different
                 start = eff_time[0].get("@value", "Unknown")
        except (KeyError, IndexError):
            pass

        start = format_date(start)

        text = f"""
Medication
Name: {med}
Route: {route}
Start Date: {start}
"""
        docs.append(make_doc(text, "MEDICATIONS", "10160-0"))
    return docs

# ---------------- PROBLEMS ----------------
def semantic_problems(section):
    docs = []
    for entry in as_list(section.get("entry")):
        obs = entry["act"]["entryRelationship"]["observation"]
        problem = obs["value"]["@displayName"]
        onset = format_date(obs["effectiveTime"]["low"]["@value"])

        text = f"""
Problem
Condition: {problem}
Onset Date: {onset}
"""
        docs.append(make_doc(text, "PROBLEMS", "11450-4"))
    return docs

# ---------------- TREATMENT PLAN ----------------
def semantic_treatment(section):
    docs = []
    for entry in as_list(section.get("entry")):
        proc = entry.get("procedure", {})
        plan = proc.get("code", {}).get("@displayName", "Unknown Plan")
        date = format_date(proc.get("effectiveTime", {}).get("@value"))

        text = f"""
Treatment Plan
Plan: {plan}
Planned On: {date}
"""
        docs.append(make_doc(text, "TREATMENT_PLAN", "18776-5"))
    return docs

# ---------------- PROCEDURES ----------------
def semantic_procedures(section):
    docs = []
    for entry in as_list(section.get("entry")):
        proc = entry.get("procedure", {})
        name = proc.get("code", {}).get("@displayName", "Unknown Procedure")
        date = format_date(proc.get("effectiveTime", {}).get("@value"))
        
        target_site = proc.get("targetSiteCode", {}).get("@displayName")
        
        # device extraction if present
        device = None
        try:
             device = proc["participant"]["participantRole"]["playingDevice"]["code"]["@displayName"]
        except (KeyError, TypeError):
             pass

        text = f"""
Procedure
Type: {name}
Date: {date}
"""
        if target_site:
            text += f"Target Site: {target_site}\n"
        if device:
            text += f"Device Used: {device}\n"

        docs.append(make_doc(text, "PROCEDURES", "47519-4"))
    return docs

# ---------------- RESULTS ----------------
def semantic_results(section):
    docs = []
    # Results are often nested: organizer -> component -> observation
    for entry in as_list(section.get("entry")):
        organizer = entry.get("organizer", {})
        for comp in as_list(organizer.get("component")):
            obs = comp.get("observation", {})
            test_name = obs.get("code", {}).get("@displayName", "Unknown Test")
            
            # Value can be @value, @displayName, or text content
            val = obs.get("value", {})
            if isinstance(val, dict):
                 value = val.get("@value") or val.get("@displayName")
                 unit = val.get("@unit")
            elif isinstance(val, list) and len(val) > 0:
                 # rare case
                 value = val[0].get("@value")
                 unit = val[0].get("@unit")
            else:
                 value = "Unknown"
                 unit = None

            date = format_date(obs.get("effectiveTime", {}).get("low", {}).get("@value"))

            text = f"""
Lab Result
Test: {test_name}
Value: {value} {unit if unit else ""}
Date: {date}
"""
            docs.append(make_doc(text, "RESULTS", "30954-2"))
    return docs

# ---------------- SOCIAL HISTORY ----------------
def semantic_social_history(section):
    docs = []
    for entry in as_list(section.get("entry")):
        obs = entry.get("observation", {})
        # Observation code (e.g. Smoking Status)
        topic = obs.get("code", {}).get("@displayName", "Social History")
        # Value (e.g. Current Smoker)
        status = obs.get("value", {}).get("@displayName", "Unknown Status")
        date = format_date(obs.get("effectiveTime", {}).get("low", {}).get("@value"))

        text = f"""
Social History
Topic: {topic}
Status: {status}
Date: {date}
"""
        docs.append(make_doc(text, "SOCIAL_HISTORY", "29762-2"))
    return docs

# ---------------- VITAL SIGNS ----------------
def semantic_vitals(section):
    docs = []
    for entry in as_list(section.get("entry")):
        organizer = entry.get("organizer", {})
        for comp in as_list(organizer.get("component")):
            obs = comp.get("observation", {})
            name = obs.get("code", {}).get("@displayName", "Vital Sign")
            val_data = obs.get("value", {})
            value = val_data.get("@value", "N/A")
            unit = val_data.get("@unit", "")
            date = format_date(obs.get("effectiveTime", {}).get("low", {}).get("@value"))

            text = f"""
Vital Sign
Measurement: {name}
Value: {value} {unit}
Date: {date}
"""
            docs.append(make_doc(text, "VITAL_SIGNS", "8716-3"))
    return docs

# ---------------- GOALS ----------------
def semantic_goals(section):
    docs = []
    for entry in as_list(section.get("entry")):
        obs = entry.get("observation", {})
        goal = obs.get("code", {}).get("@displayName", "Unknown Goal")
        # Sometimes goal value is in 'value'
        target = obs.get("value", {}).get("@value")
        unit = obs.get("value", {}).get("@unit", "")
        
        status = obs.get("statusCode", {}).get("@code", "active")
        date = format_date(obs.get("effectiveTime", {}).get("@value"))

        text = f"""
Goal
Description: {goal}
Status: {status}
Target: {target if target else "N/A"} {unit}
Date: {date}
"""
        docs.append(make_doc(text, "GOALS", "61146-7"))
    return docs

# ---------------- HEALTH CONCERNS ----------------
def semantic_health_concerns(section):
    docs = []
    # This section can be complex. We try to find the underlying problem observation.
    for entry in as_list(section.get("entry")):
        act = entry.get("act", {})
        # Concern often wraps an observation
        rels = as_list(act.get("entryRelationship"))
        for rel in rels:
            obs = rel.get("observation", {})
            if not obs: continue
            
            problem = obs.get("value", {}).get("@displayName")
            if not problem:
                 problem = obs.get("code", {}).get("@displayName")
            
            status = obs.get("statusCode", {}).get("@code")
            date = format_date(obs.get("effectiveTime", {}).get("low", {}).get("@value"))
            
            if problem:
                text = f"""
Health Concern
Concern: {problem}
Status: {status}
Date: {date}
"""
                docs.append(make_doc(text, "HEALTH_CONCERNS", "75310-3"))
    return docs

# ---------------- MENTAL STATUS ----------------
def semantic_mental_status(section):
    docs = []
    for entry in as_list(section.get("entry")):
        obs = entry.get("observation", {})
        # Sometimes it's a finding
        finding = obs.get("value", {}).get("@displayName")
        if not finding:
            finding = obs.get("code", {}).get("@displayName")
            
        date = format_date(obs.get("effectiveTime", {}).get("low", {}).get("@value"))

        text = f"""
Mental Status
Finding: {finding}
Date: {date}
"""
        docs.append(make_doc(text, "MENTAL_STATUS", "10190-7"))
    return docs

# ---------------- HELPER: TEXT EXTRACTION ----------------
def extract_text_content(section, title, loinc):
    """Fallback for sections that rely on narrative text lists."""
    text_node = section.get("text", {})
    sentences = []
    
    # Try list items
    if "list" in text_node and "item" in text_node["list"]:
        items = as_list(text_node["list"]["item"])
        sentences.extend(str(x) for x in items if x)
    elif "content" in text_node:
        c = text_node["content"]
        if isinstance(c, str):
            sentences.append(c)
        elif isinstance(c, list):
             sentences.extend(str(x) for x in c)
    elif isinstance(text_node, str):
        sentences.append(text_node)
        
    if not sentences:
        return []

    combined_text = "\n".join(sentences)
    doc_text = f"""
{title}
{combined_text}
"""
    return [make_doc(doc_text, title, loinc)]

# ---------------- REASON FOR REFERRAL ----------------
def semantic_reason_referral(section):
    # Mostly text based
    return extract_text_content(section, "REASON_FOR_REFERRAL", "42349-1")

# ---------------- ASSESSMENTS ----------------
def semantic_assessments(section):
    # Mostly text based
    return extract_text_content(section, "ASSESSMENTS", "51848-0")

# ---------------- DEMOGRAPHICS ----------------
def semantic_demographics(ccda):
    docs = []
    logger.info("Processing demographics section")
    try:
        patient_role = ccda["ClinicalDocument"]["recordTarget"]["patientRole"]
        patient = patient_role["patient"]
        
        # Name
        name_node = patient.get("name", {})
        given_list = as_list(name_node.get("given", []))
        given_parts = []
        for p in given_list:
            if isinstance(p, dict):
                given_parts.append(p.get("#text", ""))
            else:
                given_parts.append(str(p))
        given = " ".join(given_parts).strip()
        
        family_node = name_node.get("family", {})
        if isinstance(family_node, dict):
            family = family_node.get("#text", "")
        else:
            family = str(family_node)
            
        full_name = f"{given} {family}".strip()
        logger.debug(f"Extracted patient name: {full_name}")

        # Gender
        gender = patient.get("administrativeGenderCode", {}).get("@displayName", "Unknown")
        
        # DOB
        dob = format_date(patient.get("birthTime", {}).get("@value"))
        
        # Address
        addr_node = patient_role.get("addr", {})
        street = addr_node.get("streetAddressLine", "")
        city = addr_node.get("city", "")
        state = addr_node.get("state", "")
        zip_code = addr_node.get("postalCode", "")
        country = addr_node.get("country", "")
        address = f"{street}, {city}, {state} {zip_code}, {country}".strip()
        
        # Telecom (Phone)
        telecom_node = as_list(patient_role.get("telecom", []))
        telecoms = []
        for t in telecom_node:
            val = t.get("@value", "")
            use = t.get("@use", "")
            if val:
                telecoms.append(f"{val} ({use})")
        telecom_str = ", ".join(telecoms)

        # Race & Ethnicity
        race_node = as_list(patient.get("raceCode", []))
        races = [r.get("@displayName", "") for r in race_node if r.get("@displayName")]
        race = ", ".join(races)
        
        ethnicity = patient.get("ethnicGroupCode", {}).get("@displayName", "Unknown")
        
        # Language
        lang = patient.get("languageCommunication", {}).get("languageCode", {}).get("@code", "en")

        text = f'''
Patient Demographics
Patient Name: {full_name}
Gender: {gender}
Date of Birth: {dob}
Address: {address}
Telecom: {telecom_str}
Race: {race}
Ethnicity: {ethnicity}
Language: {lang}
'''
        # Using a generic LOINC or placeholder since demographics is header info
        docs.append(make_doc(text, "DEMOGRAPHICS", "N/A"))
        logger.info(f"Successfully extracted demographics for patient: {full_name}")
        
    except Exception as e:
        logger.error(f"Error extracting demographics: {e}", exc_info=True)
        print(f"Error extracting demographics: {e}")
        pass
    print(docs)
    return docs



# ---------------- MASTER PIPELINE ----------------
def build_semantic_docs(json_path):
    logger.info(f"Starting semantic document extraction from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        ccda = json.load(f)
    logger.info("Successfully loaded CCDA JSON file")

    docs = []
    
    # Extract Demographics (Header Level)
    logger.info("Extracting demographics")
    docs.extend(semantic_demographics(ccda))

    logger.info("Processing clinical sections")
    section_count = 0
    for comp in get_sections(ccda):
        section = comp["section"]
        code = section["code"]["@code"]
        section_count += 1

        if code == "48765-2":
            logger.debug("Processing ALLERGIES section")
            docs.extend(semantic_allergies(section))
        elif code == "46240-8":
            logger.debug("Processing ENCOUNTERS section")
            docs.extend(semantic_encounters(section))
        elif code == "47420-5":
            logger.debug("Processing FUNCTIONAL_STATUS section")
            docs.extend(semantic_functional_status(section))
        elif code == "11369-6":
            logger.debug("Processing IMMUNIZATIONS section")
            docs.extend(semantic_immunizations(section))
        elif code == "46264-8":
            logger.debug("Processing MEDICAL_EQUIPMENT section")
            docs.extend(semantic_equipment(section))
        elif code == "10160-0":
            logger.debug("Processing MEDICATIONS section")
            docs.extend(semantic_medications(section))
        elif code == "11450-4":
            logger.debug("Processing PROBLEMS section")
            docs.extend(semantic_problems(section))
        elif code == "18776-5":
            logger.debug("Processing TREATMENT_PLAN section")
            docs.extend(semantic_treatment(section))
        # New Sections
        elif code == "47519-4":
            logger.debug("Processing PROCEDURES section")
            docs.extend(semantic_procedures(section))
        elif code == "30954-2":
            logger.debug("Processing RESULTS section")
            docs.extend(semantic_results(section))
        elif code == "29762-2":
            logger.debug("Processing SOCIAL_HISTORY section")
            docs.extend(semantic_social_history(section))
        elif code == "8716-3":
            logger.debug("Processing VITAL_SIGNS section")
            docs.extend(semantic_vitals(section))
        elif code == "61146-7":
            logger.debug("Processing GOALS section")
            docs.extend(semantic_goals(section))
        elif code == "75310-3":
            logger.debug("Processing HEALTH_CONCERNS section")
            docs.extend(semantic_health_concerns(section))
        elif code == "42349-1":
            logger.debug("Processing REASON_FOR_REFERRAL section")
            docs.extend(semantic_reason_referral(section))
        elif code == "10190-7":
            logger.debug("Processing MENTAL_STATUS section")
            docs.extend(semantic_mental_status(section))
        elif code == "51848-0":
            logger.debug("Processing ASSESSMENTS section")
            docs.extend(semantic_assessments(section))
        else:
            logger.warning(f"Unknown section code: {code}")

    logger.info(f"Processed {section_count} sections, created {len(docs)} semantic documents")
    return docs


if __name__ == "__main__":
    logger.info("Semantic analysis script started")
    data = build_semantic_docs("./CCDA_Converted.json")
    print(json.dumps(data, indent=2))
    logger.info("Semantic analysis script completed")
