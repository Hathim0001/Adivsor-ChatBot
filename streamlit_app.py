import streamlit as st
import torch
import os
from openai import OpenAI
from streamlit_chat import message
from transformers import BertTokenizer, BertForSequenceClassification

# GitHub Token for OpenAI API via Azure Inference
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Allow users to input GitHub Token
with st.sidebar:
    st.title("Configuration")
    user_github_token = st.text_input("Enter your GitHub Token:", type="password")
    if user_github_token:
        GITHUB_TOKEN = user_github_token
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses Legal-BERT for classification and GPT-4o (via Azure AI) for risk analysis.")

# Initialize OpenAI client with Azure AI Inference
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

if GITHUB_TOKEN:
    client = OpenAI(
        base_url=endpoint,
        api_key=GITHUB_TOKEN,
    )
else:
    st.warning("⚠️ Please enter your GitHub Token in the sidebar.")

@st.cache_resource
def load_model():
    try:
        model = BertForSequenceClassification.from_pretrained("Prakarsha01/fine-tuned-legal-bert-v2")
        tokenizer = BertTokenizer.from_pretrained("Prakarsha01/fine-tuned-legal-bert-v2")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load Legal-BERT model
model, tokenizer = load_model()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model is not None:
    model.to(device)

def classify_clause_legal_bert(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.item()
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return -1

def run_gpt_integration(classification_label, risk_analysis, clause):
    if not GITHUB_TOKEN:
        return "Error: GitHub Token is missing."

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a legal assistant analyzing contract clauses."},
                {"role": "user", "content": f"Clause: {clause}\n\nClassification: {classification_label}\n\nRisks: {risk_analysis}"}
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in GPT integration: {str(e)}"

def run_riskAnalysis(clause):
    if not GITHUB_TOKEN:
        return "Error: GitHub Token is missing."

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Analyze this contract clause and list potential risks."},
                {"role": "user", "content": clause}
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in risk analysis: {str(e)}"

def classify_and_analyze_clause(clause):
    classification_result = classify_clause_legal_bert(clause)
    classification_label = "Audit Clause" if classification_result == 1 else "Not an Audit Clause"
    
    risk_analysis = run_riskAnalysis(clause)
    integrated_response = run_gpt_integration(classification_label, risk_analysis, clause)
    
    return integrated_response

# Streamlit UI
st.title("GitHub-AI Contract Clause Analyzer")
st.markdown("Enter a contract clause to classify and analyze risks.")

user_input = st.text_area("Enter clause:", "")
if st.button("Analyze"):
    if not GITHUB_TOKEN:
        st.error("Please enter a valid GitHub Token in the sidebar.")
    else:
        with st.spinner("Processing..."):
            response = classify_and_analyze_clause(user_input)
            st.success(response)
