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

def recognize_intent(user_input):

    # Use ConversationAnalysisClient to recognize intents
    response = conversation_analysis_client.analyze_conversation(
          task={
            "kind": "Conversation",
            "analysisInput": {
                "conversationItem": {
                    "id": "1",
                    "text": user_input,
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

def process_intent(intent, full_text):
    if intent == "summary":
        return summarize_text(full_text)
    elif intent == "RedactPII":
        # Implement your PII redaction logic here
        return redact_pii(full_text)
    elif intent == "GetEntities":
        # Implement your entity extraction logic here
        print("Extracting entities...")
        # Example: return some extracted entities
        return extract_entities(full_text)
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
    # ... (Your existing code for document analysis)
    # Initialize the Document Intelligence client
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=azure_endpoint, credential=AzureKeyCredential(azure_api_key)
    )

    # Begin analyzing the document's layout
    poller = document_intelligence_client.begin_analyze_document(
        model_id="prebuilt-layout",  # Replace with correct model if necessary
        analyze_request=data,
        content_type="application/octet-stream"
    )

    # Retrieve the analysis result
    result: AnalyzeResult = poller.result()

    # Check for handwritten content
    if result.styles and any(style.is_handwritten for style in result.styles):
        print("Document contains handwritten content")
    else:
        print("Document does not contain handwritten content")

    # Check if the document contains text and tables
    has_text = len(result.pages) > 0 and any(len(page.lines) > 0 for page in result.pages)
    has_tables = result.tables is not None and len(result.tables) > 0

    aggregated_text = []

    # Create a Word document
    doc = Document()
    doc.add_heading('Document Analysis Result', level=1)

    if has_text:
        for page in result.pages:
            print(f"---- Analyzing layout from page #{page.page_number}----")
            aggregated_text.append(f"Page {page.page_number}:\n")
            doc.add_heading(f'Page {page.page_number}', level=2)

            # Aggregate all lines of text before processing word by word
            page_text = []  # To hold text for the current page
            for line_idx, line in enumerate(page.lines):
                words = get_words(page, line)
                line_text = " ".join(word.content for word in words)
                print(f" ... Line # {line_idx} has text: '{line_text}'")

                # Append line text to page_text
                page_text.append(line_text)
                # Add the line to the document (optional)
                #doc.add_paragraph(line_text.strip())

            # Join all lines into a single text for the current page
            aggregated_text.append("\n".join(page_text) + "\n")
            processed_words = []
            # Now process each word for confidence checks
            for line in page.lines:
                words = get_words(page, line)
                for word in words:
                    print(f" ...... Word '{word.content}' has a confidence of {word.confidence}")

                    # Process the word and get the updated content, using the full context
                    processed_word = process_word(word, "\n".join(page_text))
                    processed_words.append(processed_word.strip())
            processed_paragraph = " ".join(processed_words)  # Join words with a space
            #doc.add_paragraph(processed_paragraph)  # Add the combined text as a paragraph
            st.text_area("Analysis Output", value=processed_paragraph, height=400)
                    # Assuming processed_words is your list of words

                    # Append the processed word to the aggregated text
            aggregated_text.append(processed_word + " ")
            #full_text = " ".join(aggregated_text)
            return processed_paragraph

    # Process tables in the document (remains unchanged)
    if has_tables:
        # Initialize a string to hold the processed table content
        table_output = ""
        for table_idx, table in enumerate(result.tables):
            print(f"Table # {table_idx} has {table.row_count} rows and {table.column_count} columns")

            # Add a table to the Word document
            word_table = doc.add_table(rows=table.row_count + 1, cols=table.column_count)
            hdr_cells = word_table.rows[0].cells
            for i in range(table.column_count):
                hdr_cells[i].text = f'Column {i + 1}'  # Optionally customize header

            for cell in table.cells:
                cell_content = cell.content
                print(f" ... Cell[{cell.row_index}][{cell.column_index}] has text '{cell_content}'")

                # Process the cell content
                processed_words = []
                words = cell_content.split()  # Split the cell content into words
                for word in words:
                    # Create a Word object for each word
                    word_obj = type('', (), {'content': word, 'confidence': 0.8})()  # Assuming 0.8 confidence
                    processed_word = process_word(word_obj, cell_content)
                    processed_words.append(processed_word)

                # Join processed words back into a string
                processed_cell_content = " ".join(processed_words)
                # Append processed content to table_output instead of Word document
                table_output += f"Row {cell.row_index + 1}, Column {cell.column_index + 1}: {processed_cell_content}\n"
                #word_table.cell(cell.row_index + 1, cell.column_index).text = processed_cell_content  # Fill in the table content
                aggregated_text.append(processed_cell_content + "\n")  # Append processed cell content
                # After processing the entire table, display it using st.text_area
                st.text_area("Processed Table Content", value=table_output, height=400)
    print("=====================================================")


    # Aggregate the results and return
    #After processing, you can handle the aggregated text
    #full_text = " ".join(aggregated_text)
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
            st.write(f"Result Text: {result_text}")  # Debug statement
            # Create two columns
            #col1, col2 = st.columns(2)
            #with col1:
                #st.text_area("Analysis Output", value=result_text, height=400)
    
            user_command = st.text_input("Enter a command (e.g., 'summary', 'RedactPII', 'GetEntities'):")
    
            if user_command:
                intent = recognize_intent(user_command)
                st.write(f"Intent Text: {intent}")  # Debug statement
                response_message = process_intent(intent, result_text)
                st.text_area("Output", value=response_message, height=400)
                #st.write(f"Command Response: {response_message}")
                #st.text_area("Output", value=response_message, height=300)


if __name__ == "__main__":
    analyze_document_app()
print("Hello, GitHub!")
