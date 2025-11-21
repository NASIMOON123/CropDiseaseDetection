🌱 Crop Disease Detection Using CNN

A simple deep-learning project that detects crop diseases from leaf images using a Convolutional Neural Network (CNN). Users can upload a leaf image, and the model predicts whether the crop is healthy or infected.

📌 Overview

This model is built using TensorFlow/Keras and trained on a dataset of plant leaf images.
The project includes a Streamlit interface for easy image uploading and prediction.

🧠 Model Details

CNN architecture with Conv2D, MaxPooling, Flatten, Dense, and Dropout layers

Trained on labeled leaf images

Model saved as .h5 (not pushed into github)

▶️ How to Run
pip install -r requirements.txt


Download the .h5 model and place it in the model/ folder.
Then run:

streamlit run streamlit_app.py

⭐ Features

Upload a leaf image

Instant disease prediction

Simple and clean UI

Lightweight and easy to run

📂 Project Structure
model/
  model.h5
  crop_disease_prediction.py
streamlit_app.py
class_map.json
disease_symptoms.json
requirements.txt
