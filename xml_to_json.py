import xml.etree.ElementTree as ET
import json
import re
import sys
import os
from logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)


def strip_namespace(tag):
    """
    Removes the namespace URI from the tag name.
    Example: {urn:hl7-org:v3}ClinicalDocument -> ClinicalDocument
    """
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def elem_to_internal(elem, strip_ns=True):
    """
    Recursive function to convert an ElementTree element into a dictionary.
    """
    d = {}

    # 1. Handle Attributes
    # We prefix attributes with '@' to distinguish them from child nodes
    if elem.attrib:
        for key, value in elem.attrib.items():
            if strip_ns:
                key = strip_namespace(key)
            d[f"@{key}"] = value

    # 2. Handle Text Content
    text = elem.text.strip() if elem.text else ""
    if text:
        # If the element has text AND attributes/children, store text in '#text'
        if elem.attrib or len(elem) > 0:
            d["#text"] = text
        else:
            # If it's just text (no attributes or children), just return the string
            # But wait, we need to return a dict to the parent to attach it.
            # We handle the "just string" case in the parent loop or return a specific structure.
            # For consistency in complex XML, we often stick to #text if mixed,
            # but for simple nodes <name>John</name>, we want {"name": "John"}.
            d["#text"] = text

    # 3. Handle Child Nodes
    for child in elem:
        tag = child.tag
        if strip_ns:
            tag = strip_namespace(tag)

        child_data = elem_to_internal(child, strip_ns)

        # Simplify: If child_data is a dict with only '#text' and no attributes,
        # unwrap it to a simple string.
        if isinstance(child_data, dict) and len(child_data) == 1 and "#text" in child_data:
            child_data = child_data["#text"]

        # Handling Lists:
        # If the tag already exists, turn it into a list (or append to existing list)
        if tag in d:
            if isinstance(d[tag], list):
                d[tag].append(child_data)
            else:
                d[tag] = [d[tag], child_data]
        else:
            d[tag] = child_data

    return d


def convert_xml_to_json(xml_file_path, json_file_path):
    logger.info(f"Starting XML to JSON conversion: {xml_file_path} -> {json_file_path}")
    print(f"Processing: {xml_file_path}...")

    if not os.path.exists(xml_file_path):
        logger.error(f"File not found: {xml_file_path}")
        print(f"Error: File not found at {xml_file_path}")
        return

    try:
        # Parse the XML file
        logger.debug(f"Parsing XML file: {xml_file_path}")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        logger.info(f"Successfully parsed XML. Root tag: {root.tag}")

        # Convert to Dictionary
        root_tag = strip_namespace(root.tag)
        logger.debug(f"Converting XML to dictionary structure. Root tag (stripped): {root_tag}")
        data = {root_tag: elem_to_internal(root)}
        logger.info("XML to dictionary conversion completed")

        # Write to JSON file
        logger.debug(f"Writing JSON data to: {json_file_path}")
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)

        logger.info(f"Successfully converted and saved to: {json_file_path}")
        print(f"Success! Converted data saved to: {json_file_path}")
        return True

    except ET.ParseError as e:
        logger.error(f"XML parsing failed for {xml_file_path}: {e}")
        print(f"Error: Failed to parse XML. The file might be corrupted.\nDetails: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during XML to JSON conversion: {e}")
        print(f"An unexpected error occurred: {e}")


# --- Execution ---
if __name__ == "__main__":
    logger.info("XML to JSON converter started")
    # Define file names
    input_xml = "CCDA_23103_20Oct2017_1043418.xml"
    output_json = "CCDA_Converted.json"

    # Create a dummy file for demonstration if it doesn't exist (using the content you provided)
    # In your real environment, you don't need this 'if' block if the file is already there.
    if not os.path.exists(input_xml):
        logger.warning(f"Input file not found: {input_xml}")
        print("Note: Input file not found in current directory. Please ensure the file exists.")
    else:
        convert_xml_to_json(input_xml, output_json)
    logger.info("XML to JSON converter finished")