import streamlit as st
from transformers import BertTokenizer, AutoModelForSequenceClassification, AutoModel, AutoImageProcessor
import torch
import torch.nn as nn
from openai import OpenAI
import requests
import easyocr
from PIL import Image
import io
import os
import time
import numpy as np

# ------------------------------
# KONFIGURASI
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os

# API GPT-5
API_TOKEN = st.secrets["API_TOKEN"] if "API_TOKEN" in st.secrets else os.getenv("API_TOKEN")
API_ENDPOINT = "https://models.github.ai/inference"
API_MODEL = "openai/gpt-4.1"

# Google Search API
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = st.secrets["SEARCH_ENGINE_ID"] if "SEARCH_ENGINE_ID" in st.secrets else os.getenv("SEARCH_ENGINE_ID")

# Screenshot API
SCREENSHOT_API_KEY = st.secrets["SCREENSHOT_API_KEY"] if "SCREENSHOT_API_KEY" in st.secrets else os.getenv("SCREENSHOT_API_KEY")

# Model Multimodal
TEXT_MODEL = 'prajjwal1/bert-mini'
IMAGE_MODEL = 'apple/mobilevit-small'

# ------------------------------    
# MODEL MULTIMODAL
# ------------------------------
class MultimodalClassifier(nn.Module):
    def __init__(self, num_labels, text_model_name, image_model_name):
        super(MultimodalClassifier, self).__init__()

        # Load text and image models
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.image_model = AutoModel.from_pretrained(image_model_name)

        # Get feature sizes from configs
        text_feature_size = self.text_model.config.hidden_size
        image_feature_size = self.image_model.config.neck_hidden_sizes[-1]

        self.fusion_dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(text_feature_size + image_feature_size, num_labels)

    def forward(self, input_ids, attention_mask, pixel_values):
        # ----- Text branch -----
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # [batch_size, text_hidden]

        # ----- Image branch -----
        image_outputs = self.image_model(pixel_values=pixel_values)

        # MobileViT gives pooler_output directly (global pooled feature)
        image_features = image_outputs.pooler_output  # [batch_size, image_hidden]

        # ----- Fusion -----
        combined_features = torch.cat((text_features, image_features), dim=1)
        fused_output = self.fusion_dropout(combined_features)
        logits = self.classifier(fused_output)

        return logits

# ------------------------------    
# FUNGSI UTILITAS
# ------------------------------
@st.cache_resource
def load_text_model():
    MODEL_PATH = "riozulfandy/finetuned-komentar-judol-indobert-base-p1"
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    return tokenizer, model

@st.cache_resource
def load_multimodal_model():
    # Load tokenizer and image processor
    text_tokenizer = BertTokenizer.from_pretrained(TEXT_MODEL)
    image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL)
    
    # Load multimodal model
    model = MultimodalClassifier(num_labels=2, text_model_name=TEXT_MODEL, image_model_name=IMAGE_MODEL)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    
    return text_tokenizer, image_processor, model

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['id', 'en'], gpu=torch.cuda.is_available())

# Load models
tokenizer, model = load_text_model()
reader = load_ocr_reader()
label_map = {0: "judi", 1: "non-judi"}

# Try to load multimodal model if available
try:
    mm_tokenizer, mm_image_processor, mm_model = load_multimodal_model()
    multimodal_available = True
except Exception as e:
    st.warning(f"Model multimodal tidak dapat dimuat: {str(e)}")
    multimodal_available = False

client = OpenAI(base_url=API_ENDPOINT, api_key=API_TOKEN)

def extract_site_name(comment_text):
    prompt = f"Ekstrak nama situs judi online dari teks komentar berikut: {comment_text}"
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Anda membantu mengekstrak informasi dari teks dengan hanya menjawab dengan nama situs judi online atau 'tidak terdapat'."},
            {"role": "user", "content": prompt} 
        ],
        model=API_MODEL
    )
    return str(response.choices[0].message.content.strip()).lower()

def google_search(query, num_results=3):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": str(query),
        "num": int(num_results),
        "hl": "id",   # Bahasa Indonesia
        "gl": "id",   # Lokasi Indonesia
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    results = []
    if "items" in data:
        for item in data["items"]:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
    return results

def take_screenshot(url):
    """Take screenshot of URL using Screenshot API"""
    screenshot_url = f"https://shot.screenshotapi.net/v3/screenshot?token={SCREENSHOT_API_KEY}&url={url}"
    response = requests.get(screenshot_url)
    data = response.json()
    png_url = data.get("screenshot")
    
    if png_url:
        # Download screenshot
        img_data = requests.get(png_url).content
        # Return image data directly
        return img_data
    return None

def extract_text_from_image(img_data):
    """Extract text from image using OCR"""
    # Convert image data to PIL Image
    image = Image.open(io.BytesIO(img_data))
    
    # Convert to numpy array for EasyOCR
    image_np = np.array(image)
    
    # Perform OCR
    ocr_result = reader.readtext(image_np, detail=1, paragraph=False)
    if ocr_result:
        extracted_text = ' '.join([res[1] for res in ocr_result])
        return extracted_text.strip()
    return ""

def classify_with_multimodal(img_data, text):
    """Classify using multimodal model"""
    if not multimodal_available:
        return None, None
        
    try:
        # Prepare text inputs
        text_inputs = mm_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        
        # Prepare image inputs
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        image_inputs = mm_image_processor(images=image, return_tensors='pt').to(device)
        
        # Get predictions
        with torch.no_grad():
            logits = mm_model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                pixel_values=image_inputs['pixel_values']
            )
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).cpu().item()
            confidence = probs[0][predicted_class].cpu().item()
            
        return label_map[predicted_class], confidence
    except Exception as e:
        st.error(f"Error dalam klasifikasi multimodal: {str(e)}")
        return None, None

# ------------------------------
# STREAMLIT APP
# ------------------------------
st.title("Prototype: Sistem Deteksi Situs Judi Online menggunakan Integrasi Auto-Crawling dan Klasifikasi Multimodal berdasarkan Komentar Media Sosial dan Tangkapan Layar")

# Daftar contoh komentar
contoh_komentar = [
    "10:12 ─═𝗕𝗘𝗥𝗞𝗔𝗛𝟵𝟵 ═─ emang paling juara nya sih",
    "gue udah ga bayangin hidup alexis17 , , udah sumber penghasilan",
    "18:26 ohhh jadi jersey kipernya warna ijoo wkwk, oke siip"
]

# Pilihan contoh komentar
selected_example = st.selectbox("Pilih contoh komentar:", ["(Manual input)"] + contoh_komentar)

if selected_example != "(Manual input)":
    user_input = selected_example
else:
    user_input = st.text_area("Masukkan komentar:")

top_k = st.selectbox("Pilih jumlah hasil pencarian (Top-K):", [1, 2, 3], index=0)

# Tombol Prediksi
if st.button("Prediksi"):
    if user_input.strip():
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).cpu().item()
            confidence = probs[0][predicted_class].cpu().item()

        st.session_state.predicted_label = label_map[predicted_class]
        st.session_state.confidence = confidence

        st.subheader("Hasil Prediksi")
        st.write(f"Label: **{st.session_state.predicted_label}**")
        st.write(f"Akurasi keyakinan: **{confidence:.4f}**")
        st.progress(confidence)

        if st.session_state.predicted_label == "judi":
            with st.spinner("Mengambil nama situs dari komentar..."):
                site_name = extract_site_name(user_input)
            st.session_state.site_name = site_name
            st.success(f"Nama situs terdeteksi: {site_name}")
    else:
        st.error("Masukkan komentar terlebih dahulu!")

# Tombol Cari Google
if (
    "predicted_label" in st.session_state
    and "site_name" in st.session_state
    and st.session_state.predicted_label == "judi"
    and st.session_state.site_name != "tidak terdapat"
):
    with st.spinner(f"Mencari '{st.session_state.site_name}' di Google..."):
        search_results = google_search(st.session_state.site_name, num_results=top_k)
        st.session_state.search_results = search_results

    if search_results:
        st.subheader("Hasil Pencarian Google")
        for i, res in enumerate(search_results, 1):
            st.markdown(f"**{i}. [{res['title']}]({res['link']})**")
            st.write(res['snippet'])
            
            # Capture screenshot and extract text if enabled
            with st.spinner(f"Mengambil screenshot dan ekstraksi teks {i}/{len(search_results)}..."):
                img_data = take_screenshot(res['link'])
                
                if img_data:
                    # Display screenshot
                    st.subheader(f"Screenshot {i}")
                    image = Image.open(io.BytesIO(img_data))
                    st.image(image, caption=f"Screenshot dari {res['link']}", use_container_width=True)
                    
                    # Extract and display text
                    extracted_text = ""
                    extracted_text = extract_text_from_image(img_data)
                    if extracted_text:
                        st.subheader(f"Teks yang Diekstrak {i}")
                        st.text_area(f"Teks dari {res['title']}", extracted_text, height=150)
                    else:
                        st.info("Tidak ada teks yang dapat diekstrak dari gambar")
                    
                    # Multimodal classification if enabled
                    if multimodal_available and extracted_text:
                        with st.spinner("Melakukan klasifikasi multimodal..."):
                            mm_label, mm_confidence = classify_with_multimodal(img_data, extracted_text)
                            
                        if mm_label is not None:
                            st.subheader("Hasil Klasifikasi Multimodal")
                            st.write(f"Label: **{mm_label}**")
                            st.write(f"Akurasi keyakinan: **{mm_confidence:.4f}**")
                            st.progress(mm_confidence)
                            
                            # Add warning if it's classified as gambling site
                            if mm_label == "judi":
                                st.warning("⚠️ Terdeteksi sebagai situs judi online berdasarkan analisis multimodal!")
                else:
                    st.error(f"Gagal mengambil screenshot untuk {res['link']}")
        
            st.write("---")
    else:
            st.warning("Tidak ada hasil ditemukan.")