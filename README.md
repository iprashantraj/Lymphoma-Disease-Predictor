# Lymphoma Disease Predictor 🔬

A web application for the classification of lymphoma subtypes from microscopy images. Built with **Streamlit**.

## 🚀 Live Demo
Access the app here: https://lymphoma-disease-predictor.streamlit.app/

## 🌟 Key Features
- **High-Accuracy Classification**: Predicts between 4 lymphoma stages: `Pre`, `Benign`, `Pro`, and `Early`.
- **Interactive Dashboard**: Clean, professional UI with real-time feedback.
- **Curated Demo Gallery**: Includes a "Quick Demo" section with the 12 most-confidently-correct images from the dataset for instant testing.
- **Detailed Metrics**: Displays class probabilities and confidence scores for every prediction.

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/iprashantraj/Lymphoma-Disease-Predictor.git
   cd Lymphoma-Disease-Predictor
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 🧠 Model Information
- **Algorithm**: Random Forest Classifier
- **Features**: Raw flattened pixel data (64x64 BGR)
- **Normalization**: Pixel values scaled to [0, 1]
- **Base Accuracy**: ~89% (Validated on ML Case Study dataset)

## 📁 Project Structure
```text
├── app.py                   # Streamlit application
├── random_forest_model.pkl  # Trained Random Forest model
├── requirements.txt         # Python dependencies
├── samples/                 # Curated demo images (3 per class)
└── BMLcasestudy.ipynb       # Original analysis and training notebook
```

---
*Developed for the ML Case Study (Lymphoma Subtype Detection).*
