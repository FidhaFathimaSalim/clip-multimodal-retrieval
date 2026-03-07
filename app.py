import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import glob

# ---------------------------------------------------------
# UI Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CLIP Multimodal Retrieval Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .gradient-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

DATASET_DIR = "dataset_images"
os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------------------------------------------------
# Core Model Pipeline
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the CLIP model and processor into memory."""
    model_id = "openai/clip-vit-base-patch32"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor, device

# Initialize the model 
model, processor, device = load_model()

def normalize_embeddings(features):
    """Normalize the embeddings to prepare for cosine similarity."""
    return features / features.norm(dim=-1, keepdim=True)

def encode_image(image):
    """Process an image and extract its normalized CLIP embedding."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return normalize_embeddings(image_features)

def encode_text(text):
    """Process a text prompt and extract its normalized CLIP embedding."""
    if isinstance(text, str):
        text = [text]
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return normalize_embeddings(text_features)

def compute_similarity(image_features, text_features):
    """
    Compute similarity between text and image embeddings.
    Applies CLIP logit scale and softmax for probabilities.
    """
    # Use CLIP logit scale multiplier uniformly
    logit_scale = model.logit_scale.exp()
    
    # Compare similarity via CLIP logits computation
    logits = (image_features @ text_features.T) * logit_scale
    
    # Flatten based on dimensionality for Softmax compatibility
    if logits.dim() == 2:
        if logits.shape[0] == 1:
            logits_1d = logits.squeeze(0)
        elif logits.shape[1] == 1:
            logits_1d = logits.squeeze(1)
        else:
            logits_1d = logits
    else:
        logits_1d = logits
        
    # Apply softmax to convert scores to probabilities natively
    probabilities = logits_1d.softmax(dim=0)
    
    return logits_1d, probabilities

@st.cache_resource(show_spinner=False)
def precompute_dataset_embeddings(dataset_path):
    """
    Load images from dataset_images, generate CLIP embeddings,
    and safely store them into memory for search logic optimization.
    """
    valid_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_path, ext)))
        image_paths.extend(glob.glob(os.path.join(dataset_path, ext.upper())))
        
    image_paths = list(set(image_paths))
    
    if len(image_paths) == 0:
        return [], None
        
    embeddings_list = []
    valid_paths = []
    
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            emb = encode_image(img)
            embeddings_list.append(emb)
            valid_paths.append(path)
        except Exception:
            pass
            
    if not embeddings_list:
        return [], None
        
    # Store all matched components mapped inside a tensor stack
    dataset_features = torch.cat(embeddings_list, dim=0)
    return valid_paths, dataset_features

# Enforce processing embeddings seamlessly into application load
with st.spinner("Caching dataset embeddings ahead of time for fast retrieval..."):
    precomputed_paths, precomputed_features = precompute_dataset_embeddings(DATASET_DIR)

def text_to_image_retrieval(query, image_paths, dataset_features, top_k=5):
    """Search and rank dataset images mapped contextually against a text inquiry."""
    # Convert query into natural language prompt seamlessly
    prompt = f"a photo of {query}"
    text_features = encode_text(prompt)
    
    # Apply similarity constraints linking the query to available images
    _, probabilities = compute_similarity(dataset_features, text_features)
    
    # Rank images via descending similarity index
    sorted_indices = torch.argsort(probabilities, descending=True)
    top_indices = sorted_indices[:top_k]
    
    results = []
    for idx in top_indices:
        idx_val = idx.item()
        confidence = probabilities[idx_val].item() * 100 # Derive % confidence scale
        results.append({
            "path": image_paths[idx_val],
            "confidence": confidence
        })
        
    return results, prompt

def image_to_text_classification(image, labels):
    """Rank multiple labels aligned against an image mapping index."""
    image_features = encode_image(image)
    text_features = encode_text(labels)
    
    _, probabilities = compute_similarity(image_features, text_features)
    
    return probabilities.detach().cpu().numpy()

# ---------------------------------------------------------
# UI Implementation
# ---------------------------------------------------------
st.markdown('<div class="gradient-header">CLIP Multimodal Retrieval System</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🖼️ Image → Text Classification", "🔍 Text → Image Retrieval"])

# --- TAB 1: Image to Text ---
with tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### Input Image")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(uploaded_image, caption="Provided Image", use_container_width=True)
            
    with col2:
        st.markdown("#### Candidate Labels")
        labels_input = st.text_input("Enter possible classifications (comma-separated)", value="dog, cat, car, airplane")
        if st.button("Predict 🚀", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please upload an image first.")
            else:
                with st.spinner("Classifying image across semantics..."):
                    labels = [lbl.strip() for lbl in labels_input.split(",") if lbl.strip()]
                    probs = image_to_text_classification(uploaded_image, labels)
                    
                    top_idx = int(np.argmax(probs))
                    st.success(f"**Top Match:** {labels[top_idx]}\n\n**Confidence:** {probs[top_idx]*100:.1f}%")
                    
                    st.markdown("##### Prediction Probabilities")
                    chart_df = pd.DataFrame({
                        "Labels": labels,
                        "Confidence (%)": probs * 100
                    }).set_index("Labels")
                    st.bar_chart(chart_df, y="Confidence (%)", color="#764ba2")


# --- TAB 2: Text to Image ---
with tab2:
    st.markdown("#### Database Semantic Lookup")
    query = st.text_input("Enter a semantic query text", placeholder="car")
    if st.button("Search Image Dataset 🔍", type="primary", use_container_width=True):
        if precomputed_paths is None or len(precomputed_paths) == 0:
            st.error("No images found in dataset_images folder")
        elif not query:
            st.warning("Please enter a text query to search our embeddings cache.")
        else:
            with st.spinner("Ranking results based on similarity..."):
                results, prompt = text_to_image_retrieval(query, precomputed_paths, precomputed_features)
                
                st.markdown(f"### Top Matches for \"{prompt}\"")
                
                # Results Grid representation dynamically configured based on counts
                st.markdown("##### Media References")
                cols = st.columns(len(results) if len(results) > 0 else 1)
                for i, result in enumerate(results):
                    with cols[i]:
                        img = Image.open(result['path']).convert("RGB")
                        st.image(img, use_container_width=True)
                        filename = os.path.basename(result['path'])
                        st.markdown(f"**{filename}**\n\nConfidence: {result['confidence']:.1f}%")
