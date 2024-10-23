# Your code goes here
# Your code goes here
import streamlit as st
import os
from io import BytesIO  # To handle file upload
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.textanalytics import TextAnalyticsClient
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from azure.ai.language.conversations import ConversationAnalysisClient
from openai import AzureOpenAI
import numpy as np
import re

# Helper functions and API clients here...
# Fetch secret keys from secret storage
azure_api_key = st.secrets['AZURE_API_KEY']
azure_endpoint = st.secrets['AZURE_ENDPOINT']
azure_openai_endpoint = st.secrets['AZURE_OPENAI_ENDPOINT']  # Assuming you store this
azure_openai_key = st.secrets['AZURE_OPENAI_KEY']  # Assuming you store this
text_analytics_api_key = st.secrets['TEXT_ANALYTICS_API_KEY']  # Assuming you store this
text_analytics_endpoint = st.secrets['TEXT_ANALYTICS_ENDPOINT']  # Assuming you store this
convers_analysis_api_key = st.secrets['CONVERSATION_ANALYSIS_API_KEY']  # Assuming you store this
convers_analysis_endpoint = st.secrets['CONVERSATION_ANALYSIS_ENDPOINT']  # Assuming you store this

# Initialize Azure OpenAI client
openai_client = AzureOpenAI(azure_endpoint=azure_openai_endpoint, api_key=azure_openai_key, api_version="2024-08-01-preview")

# Initialize Azure Text Analytics client
text_analytics_client = TextAnalyticsClient(
    endpoint=text_analytics_endpoint,
    credential=AzureKeyCredential(text_analytics_api_key)
)
conversation_analysis_client = ConversationAnalysisClient(
    convers_analysis_endpoint,AzureKeyCredential(convers_analysis_api_key)
)

# Helper function to get words within a line's spans
def get_words(page, line):
    result = []
    for word in page.words:
        if _in_span(word, line.spans):
            result.append(word)
    return result
response = None  # Initialize response

# Helper function to check if a word is within any of the spans
def _in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (
            word.span.offset + word.span.length
        ) <= (span.offset + span.length):
            return True
    return False

def recognize_intent(user_command):
    # Use ConversationAnalysisClient to recognize intents
    response = conversation_analysis_client.analyze_conversation(
          task={
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "id": "1",
                    "text": user_command,
                    "participantId": "1"
                }
            },
            "parameters": {
                "projectName": "ConvUnder",  # Replace with your project name
                "deploymentName": "Conversationn"  # Replace with your deployment name
            }
        }

    )

    intents = response["result"]["prediction"]["topIntent"]
    return intents

def process_intent(intent, result_text):
    if intent == "summary":
        return summarize_text(result_text)
    elif intent == "RedactPII":
        # Implement your PII redaction logic here
        return redact_pii(result_text)
    elif intent == "GetEntities":
        # Implement your entity extraction logic here
        print("Extracting entities...")
        # Example: return some extracted entities
        return extract_entities(result_text)
    else:
        return "Sorry, I couldn't recognize the intent."

def summarize_text(text):
    prompt = f"Please summarize the following text:\n\n{text}\n\nSummary:"

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.5,
        max_tokens=150,
    )

    summary = response.choices[0].message.content.strip()
    return summary

def redact_pii(text):
    # Example patterns for PII (you may need to adjust these based on your requirements)
    patterns = {
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b': '[REDACTED NAME]',  # Names (e.g., John Doe)
        r'\b\d{3}-\d{2}-\d{4}\b': '[REDACTED SSN]',  # Social Security Numbers
        r'\b\w+@\w+\.\w+\b': '[REDACTED EMAIL]',  # Emails
        r'\b\d{3}-\d{3}-\d{4}\b': '[REDACTED PHONE]',  # Phone Numbers
    }

    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)

    return text

def extract_entities(text):
    response = text_analytics_client.recognize_entities(documents=[text])[0]

    if not response.is_error:
        entities = [(entity.text, entity.category) for entity in response.entities]
        return entities
    else:
        return "Error in entity extraction."

# Load a sentence transformer model (you can change this to another model if needed)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to compute similarity
def compute_similarity(word1, word2):
    embeddings = model.encode([word1, word2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def process_word(word, context, file_path=None):
    response = None  # Initialize response with None or a default value
    suggested_word = word.content  # Default to the original word if no response

    # If confidence is less than 0.9, predict the word using context
    if word.confidence < 0.9:
        prompt = f"The word '{word.content}' might be incorrect. Suggest a more accurate word, considering it might be slightly distorted or misread. Only suggest if it's reasonably certain, otherwise, just return the original word. Context: {context}"

        # Prepare the request
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # If an image is provided, you would include that in the request
        if file_path:
            messages.append({"role": "system", "content": f"Image: {file_path}"})

        try:
            # Make the API call
            response = openai_client.chat.completions.create(
                model="gpt-4",  # Replace with your Azure OpenAI model deployment name
                messages=messages,
                temperature=0.7,
                max_tokens=150,
            )

            # Ensure response is valid before attempting to access choices
            if response and hasattr(response, "choices") and response.choices:
                suggested_word = response.choices[0].message.content.strip()
            else:
                st.error("No valid response received from OpenAI API.")
        except Exception as e:
            st.error(f"Error in processing OpenAI request: {e}")
            response = None  # Ensure response is handled in case of an error

        # Compute similarity
        similarity = compute_similarity(word.content, suggested_word)
        print(f" ...... Similarity between '{word.content}' and '{suggested_word}': {similarity}")

        # Add logic based on similarity
         # If the similarity is low, show both original and suggested word
        if similarity < 0.85:
            return f"{word.content} <{suggested_word}>"  # Show original + suggested word with angle brackets
        else:
            # If the similarity is high, show the original word with strikethrough for the incorrect word
            return word.content  # Show original word with strikethrough + suggested word
    else:
        return word.content


def analyze_layout(file_path):
    
    # Read the file and analyze it (similar to your original function)
    with open(file_path, 'rb') as file:
        data = file.read()

    # Call Document Intelligence API and analyze the document layout
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=azure_endpoint, credential=AzureKeyCredential(azure_api_key)
    )
    poller = document_intelligence_client.begin_analyze_document(
        model_id="prebuilt-layout",  # Replace with correct model if necessary
        analyze_request=data,
        content_type="application/octet-stream"
    )
    result: AnalyzeResult = poller.result()

    # Check for handwritten content
    if result.styles and any(style.is_handwritten for style in result.styles):
        print("Document contains handwritten content")
    else:
        print("Document does not contain handwritten content")

    # Check if the document contains text and tables
    has_text = len(result.pages) > 0 and any(len(page.lines) > 0 for page in result.pages)
    has_tables = result.tables is not None and len(result.tables) > 0

    aggregated_text1 = []
    aggregated_text2 = []
    if has_text:
        for page in result.pages:
            aggregated_text1.append(f"Page {page.page_number}:\n")
            page_text = []  # To hold text for the current page
            for line_idx, line in enumerate(page.lines):
                words = get_words(page, line)
                line_text = " ".join(word.content for word in words)
                page_text.append(line_text)

            #aggregated_text1.append("\n".join(page_text) + "\n")
            processed_words = []
            for line in page.lines:
                words = get_words(page, line)
                for word in words:
                    processed_word = process_word(word, "\n".join(page_text))
                    processed_words.append(processed_word.strip())
            processed_paragraph = " ".join(processed_words)  # Join words with a space
            aggregated_text1.append(processed_paragraph + " ")
            return " ".join(aggregated_text1)  # Return the combined text as a string
    if has_tables:
        table_output = ""
        for table_idx, table in enumerate(result.tables):
            for cell in table.cells:
                cell_content = cell.content
                processed_words = []
                words = cell_content.split()  # Split the cell content into words
                for word in words:
                    word_obj = type('', (), {'content': word, 'confidence': 0.8})()  # Assuming 0.8 confidence
                    processed_word = process_word(word_obj, cell_content)
                    processed_words.append(processed_word)
                
                processed_cell_content = " ".join(processed_words)
                table_output += f"Row {cell.row_index + 1}, Column {cell.column_index + 1}: {processed_cell_content}\n"
                aggregated_text2.append(processed_cell_content + "\n")
                return " ".join(aggregated_text2)  # Return the combined text as a string
    #return " ".join(aggregated_text)  # Return the combined text as a string
    # Aggregate the results and return
    #After processing, you can handle the aggregated text
    #full_text = " ".join(aggregated_text1)
    #return full_text

def analyze_document_app():
    response = None  # Initialize response
    st.title("Intelligent Document Processing System (IDPS)")

    uploaded_file = st.file_uploader("Upload a file for analysis", type=['jpg', 'png', 'pdf'])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        file_path = "/tmp/uploaded_file." + uploaded_file.name.split('.')[-1]

        with open(file_path, 'wb') as f:
            f.write(file_bytes)

        st.success(f"File {uploaded_file.name} uploaded successfully.")

        if st.button('Run Analysis'):
            st.write("Running analysis on the uploaded file...")

            result_text = analyze_layout(file_path)
            response_message = "No command entered."

            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Analysis Output", value=result_text, height=400)

            user_command = st.text_input("Enter a command (e.g., 'summary', 'RedactPII', 'GetEntities'):")

            if user_command:
                if user_command.strip():
                    intent = recognize_intent(user_command)
                    st.write(f"Intent Text: {intent}")  # Debug statement
                    if intent:
                        response_message = process_intent(intent, result_text)
                        with col2:
                            st.text_area("Output", value=response_message, height=400)
                else:
                    st.error("The command input cannot be empty. Please enter a valid command.")

if __name__ == "__main__":
    analyze_document_app()
