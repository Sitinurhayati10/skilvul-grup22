import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import SVC
from PIL import Image
from io import BytesIO
import base64

# Fungsi untuk memuat gambar sebagai base64
def load_image_as_base64(image_path):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Fungsi untuk pemrosesan teks
def lowercase(text):
    return text.lower()

def removepunc(text):
    import string
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_sw(text):
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def stem_text(text):
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    return ' '.join([porter.stem(word) for word in text.split()])

# Set the background colors, text color, and fonts
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    body {
        background-color: #87CEEB; /* Sky blue background */
        margin: 0; /* Remove default margin for body */
        padding: 0; /* Remove default padding for body */
        font-family: 'Poppins', sans-serif; /* Set global font to Poppins */
    }
    .st-bw {
        background-color: #ffffff; /* White background for widgets */
        border-radius: 10px; /* Add rounded corners */
        padding: 20px; /* Add padding */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Add subtle shadow */
    }
    .st-cq {
        background-color: #f7f7f7; /* Light gray background for input */
        border-radius: 10px; /* Add rounded corners */
        padding: 8px 12px; /* Add padding for input text */
        color: black; /* Set text color */
        font-family: 'Poppins', sans-serif; /* Set font to Poppins */
    }
    .st-cx {
        background-color: #f0f0f0; /* Light gray background for messages */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add padding */
        font-family: 'Poppins', sans-serif; /* Set font to Poppins */
    }
    .sidebar .block-container {
        background-color: #87CEEB; /* Sky blue background for sidebar */
        border-radius: 10px; /* Add rounded corners */
        padding: 10px; /* Add some padding for spacing */
    }
    .top-right-image-container {
        position: fixed;
        top: 30px;
        right: 0;
        padding: 20px;
        background-color: #87CEEB; /* Sky blue background for image container */
        border-radius: 0 0 0 10px; /* Add rounded corners to bottom left */
    }
    .custom-font {
        color: #444444; /* Set font color to dark gray */
        font-family: 'Poppins', sans-serif; /* Set font to Poppins */
    }
    .headline {
        color: #444444; /* Set font color to dark gray */
        font-family: 'Poppins', sans-serif; /* Set font to Poppins */
        font-size: 24px;
        font-weight: 300;
        margin-top: 0;
    }
    .title {
        color: #87CEEB; /* Light blue color for title */
        font-family: 'Poppins', sans-serif; /* Set font to Poppins */
        font-size: 32px;
        font-weight: 600;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the logo image
logo_base64 = load_image_as_base64("BookGenie.png")

st.markdown(
    f"""
    <div style='display: flex; align-items: center; gap: 15px;' class='custom-font'>
        <img src='data:image/png;base64,{logo_base64}' width='50'>
        <h1 class='title'>BookGenie</h1>
    </div>
    <p class='headline'>Predict the genre of any book with our AI technology!</p>
    """,
    unsafe_allow_html=True
)

# Kolom input teks untuk deskripsi buku
st.markdown("### Enter book summary:")
book_description = st.text_area("", height=200)

if st.button("Predict"):
    if not book_description:
        st.warning("Please enter a book summary.")
    else:
        st.info("Predicting...")

        # Load model SVM dan vectorizer
        with open("C:\\Skilvul Grup 22 Education\\bookgenie-genreprediction\\svm_model (2).pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        
        with open("C:\\Skilvul Grup 22 Education\\bookgenie-genreprediction\\tfidf_vectorizer.pkl", 'rb') as file:
            tfidf = pickle.load(file)
        
        # Memuat LabelEncoder
        with open("label_encoder.pkl", 'rb') as encoder_file:
            loaded_encoder = pickle.load(encoder_file)
        
        # Membaca data X_train
        X_train = pd.read_csv("C:\\Skilvul Grup 22 Education\\bookgenie-genreprediction\\X_train.csv")  # Ubah sesuai dengan lokasi yang benar

        # Menerapkan pemrosesan teks pada data X_train
        X_train['Combined_Text'] = X_train['Combined_Text'].apply(lowercase)
        X_train['Combined_Text'] = X_train['Combined_Text'].apply(removepunc)
        X_train['Combined_Text'] = X_train['Combined_Text'].apply(remove_sw)
        X_train['Combined_Text'] = X_train['Combined_Text'].apply(stem_text)

        # Membuat dan melatih tfidf vectorizer dari data X_train
        tfidf = TfidfVectorizer(max_features=5000)  # Atur max_features sesuai kebutuhan
        X_train_tfidf = tfidf.fit_transform(X_train['Combined_Text']).toarray()
        
        # Preprocessing deskripsi buku
        book_description_processed = [stem_text(remove_sw(removepunc(lowercase(book_description))))]

        # Transformasi deskripsi buku menggunakan TfidfVectorizer yang sama
        book_description_tfidf = tfidf.transform(book_description_processed).toarray()

        # Prediksi genre buku
        predictions = loaded_model.predict(book_description_tfidf)

        # Map hasil prediksi ke genre yang sesuai
        genre_mapping = {
            0: "adventure",
            1: "crime",
            2: "fantasy",
            3: "learning",
            4: "romance",
            5: "thriller"
        }
        
        predicted_genre = genre_mapping[predictions[0]]

        # Tampilkan hasil prediksi
        st.write("Prediction Results:")
        st.title(predicted_genre)
        st.success("Prediction completed.")
