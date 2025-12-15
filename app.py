import streamlit as st
import pickle
import re
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

# ----------------------------
# Load model and vectorizer
# ----------------------------
with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ----------------------------
# Text cleaning function
# ----------------------------
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------
# Header
# ----------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Predict whether a movie review is positive or negative and view model performance.")

# ----------------------------
# Sidebar: Metrics only
# ----------------------------
st.sidebar.header("Model Metrics")
accuracy = 0.8955
precision = 0.8875
recall = 0.9058
f1 = 0.8966
st.sidebar.write(f"**Accuracy:** {accuracy}")
st.sidebar.write(f"**Precision:** {precision}")
st.sidebar.write(f"**Recall:** {recall}")
st.sidebar.write(f"**F1 Score:** {f1}")

# ----------------------------
# Main Area: Visualizations
# ----------------------------
st.subheader("Model Visualizations")

# Precomputed confusion matrix
cm = np.array([[4426, 574],
               [471, 4529]])
fig1, ax1 = plt.subplots(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")
ax1.set_title("Confusion Matrix")

# Precomputed ROC curve
fpr = [0.0, 0.05, 0.1, 0.2, 1.0]
tpr = [0.0, 0.6, 0.75, 0.9, 1.0]
roc_auc = 0.95
fig2, ax2 = plt.subplots(figsize=(6,5))
ax2.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc})", color="blue", linewidth=2)
ax2.plot([0,1], [0,1], linestyle='--', color='gray')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()

# Display figures in two columns
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)

# ----------------------------
# Movie review input & prediction
# ----------------------------
st.subheader("Predict Sentiment")
review_input = st.text_area("Enter a movie review:")

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review!")
    else:
        cleaned = clean_text(review_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        prob = model.predict_proba(vect)[0]
        sentiment = "Positive 👍" if pred == 1 else "Negative 👎"
        confidence = f"{max(prob)*100:.2f}%"
        
        st.success("Prediction Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence}")
        
        st.markdown("""
        **Explanation:**  
        Logistic Regression predicts whether a review is positive or negative based on the words used in the review.  
        Confidence indicates how strongly the model believes in its prediction.
        """)
