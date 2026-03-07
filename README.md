# CLIP Multimodal Retrieval System 🔍

A Streamlit web application that leverages OpenAI's CLIP (Contrastive Language-Image Pre-Training) model to perform multimodal tasks such as **Zero-Shot Image Classification** and **Semantic Text-to-Image Retrieval**.

## 🌟 Features

*   **🖼️ Image → Text Classification (Zero-Shot)**:
    Upload an image and provide a comma-separated list of candidate labels. The CLIP model will predict the best matching label for the image without any prior task-specific training.
*   **🔍 Text → Image Retrieval**:
    Perform semantic search across a local database of images. The app precomputes embeddings for all images within the `dataset_images/` folder and allows you to retrieve the most relevant images based on a natural language text query (e.g., "a photo of a car").

## 🛠️ Technology Stack

*   **Python**: Core programming language.
*   **[Streamlit](https://streamlit.io/)**: Simple and interactive web UI.
*   **[PyTorch](https://pytorch.org/)**: Underlying deep learning framework.
*   **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)**: To load and run the `openai/clip-vit-base-patch32` model.

## 📁 Repository Structure

```text
├── app.py               # Main Streamlit application script
├── requirements.txt     # Python dependencies
├── dataset_images/      # Folder containing images for Text -> Image Retrieval
└── README.md            # Project documentation
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/clip-multimodal-retrieval.git
cd clip-multimodal-retrieval
```

### 2. Create a Virtual Environment (Optional but recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Provide Dataset Images
For the **Text → Image Retrieval** feature to work, place your collection of `.jpg`, `.jpeg`, or `.png` images inside the `dataset_images/` directory.

### 5. Run the Application
Start the Streamlit server:
```bash
streamlit run app.py
```
The application will open automatically in your default web browser at `http://localhost:8501`.

## 💡 How it works under the hood

1.  **Embeddings:** The CLIP model consists of an Image Encoder and a Text Encoder. It encodes both images and text text into the same multi-dimensional dense vector space.
2.  **Cosine Similarity:** To find the relationship between an image and a text prompt, the application computes the cosine similarity (scaled into probabilities via softmax) between their corresponding feature embeddings. 
3.  **Caching:** Image embeddings in the `dataset_images` folder are cached into memory at startup to ensure extremely fast search processing.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
