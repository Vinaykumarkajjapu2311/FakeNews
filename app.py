# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import warnings
from typing import List

# Disable warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings("ignore")

# Load model and tokenizer
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-small", 
        num_labels=2
    ).to(device)
    
    # Load your trained weights
    model.load_state_dict(torch.load("deberta_best.pt", map_location=device))
    
    # Set up label encoder
    le = LabelEncoder()
    le.classes_ = np.array(['fake', 'real'])  # Must match your training labels
    
    return model, tokenizer, le, device

# Prediction function
def predict(text: str, model, tokenizer, device) -> np.ndarray:
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

# Explanation function
def explain_prediction(text: str, model, tokenizer, le, device, num_features: int = 10):
    explainer = LimeTextExplainer(class_names=le.classes_)
    
    def predictor(texts: List[str]) -> np.ndarray:
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    
    exp = explainer.explain_instance(
        text, 
        predictor, 
        num_features=num_features, 
        num_samples=500,
        top_labels=len(le.classes_)  # Ensure all labels are explained
    )
    return exp

def main():
    # Initialize Streamlit
    st.set_page_config(page_title="Fake News Detector", layout="wide")
    st.title("🔍 Fake News Detection with Explainable AI")
    
    # Load model
    try:
        model, tokenizer, le, device = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return
    
    # Input form
    with st.form("news_form"):
        news_text = st.text_area(
            "Enter news text to analyze:", 
            "India records yet another single-day rise of over 28000 new cases..."
        )
        submitted = st.form_submit_button("Analyze")

    if submitted and news_text:
        with st.spinner("Analyzing text..."):
            try:
                # Get prediction
                probs = predict(news_text, model, tokenizer, device)
                pred_class = le.classes_[np.argmax(probs)]
                confidence = np.max(probs)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Results")
                    st.metric("Prediction", pred_class, f"{confidence:.1%} confidence")
                    
                    # Progress bars
                    st.write("Class Probabilities:")
                    for i, class_name in enumerate(le.classes_):
                        progress_value = int(probs[i] * 100)
                        st.progress(progress_value, text=f"{class_name}: {probs[i]:.1%}")
                
                with col2:
                    st.subheader("Model Explanation")
                    try:
                        exp = explain_prediction(news_text, model, tokenizer, le, device)
                        
                        # Show explanation for each class
                        tabs = st.tabs([f"Features for '{cls}'" for cls in le.classes_])
                        
                        for i, tab in enumerate(tabs):
                            with tab:
                                try:
                                    # Get available labels from explanation
                                    if i in exp.available_labels():
                                        weights = exp.as_list(label=i)
                                        features, scores = zip(*weights)
                                        colors = ['green' if score > 0 else 'red' for score in scores]
                                        
                                        fig, ax = plt.subplots(figsize=(8, 4))
                                        ax.barh(features, scores, color=colors)
                                        ax.set_title(f"Features contributing to '{le.classes_[i]}'")
                                        st.pyplot(fig)
                                        plt.close()
                                    else:
                                        st.warning(f"No features found for class '{le.classes_[i]}'")
                                except Exception as e:
                                    st.error(f"Couldn't explain for class {le.classes_[i]}: {str(e)}")
                    except Exception as e:
                        st.error(f"Explanation failed: {str(e)}")
            
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()