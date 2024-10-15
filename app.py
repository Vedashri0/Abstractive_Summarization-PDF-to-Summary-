import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import PyPDF2

# Load the trained model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Function to summarize text
def generate_summary(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        # Use the UploadedFile object directly without opening it
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Streamlit app
st.title("PDF Summarizer")
st.write("Upload a PDF file to get a summary.")

# File uploader
pdf_file = st.file_uploader("Choose a PDF file", type='pdf')

if pdf_file is not None:
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_file)
    
    if text:
        st.subheader("Original Text:")
        st.text_area("Text from PDF", text, height=300)

        # Generate summary
        try:
            summary = generate_summary(text)
            st.subheader("Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
    else:
        st.write("No text found in the PDF.")
