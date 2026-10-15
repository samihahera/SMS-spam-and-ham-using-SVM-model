# SMS Spam Classification using SVM

## 📖 Overview
This project is a Natural Language Processing (NLP) and Machine Learning application that classifies SMS messages as **Spam** (unwanted, promotional, or fraudulent) or **Ham** (normal, legitimate messages). It uses a Linear Support Vector Machine (SVM) model trained on the SMS Spam Collection Dataset.

## 📊 Dataset
The dataset used is the **SMS Spam Collection Dataset** from Kaggle.
- Link: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- It contains 5,572 SMS messages labeled as `ham` or `spam`.

## 🛠️ Tools and Technologies
- **Language:** Python 3
- **Environment:** VS Code & Jupyter Notebook
- **Libraries:**
  - `pandas` & `numpy` (Data manipulation)
  - `nltk` (Text preprocessing: stopwords, lemmatization)
  - `scikit-learn` (TF-IDF Vectorization, LinearSVC, Pipeline, GridSearchCV)
  - `matplotlib` & `seaborn` (Data visualization)
  - `wordcloud` (Visualizing common words in Spam vs Ham)
  - `joblib` (Saving the trained model)

## 🚀 Project Workflow
1. **Data Loading:** Loaded the CSV file and dropped unnecessary columns.
2. **Exploratory Data Analysis (EDA):** Visualized the class balance and generated Word Clouds for Spam and Ham.
3. **Data Preprocessing:** 
   - Lowercasing, removing URLs, emails, numbers, and punctuation.
   - Removing stopwords and applying Lemmatization (NLTK).
   - Removing duplicate messages.
4. **Feature Extraction:** Converted text to numerical features using `TfidfVectorizer` (max_features=5000, ngram_range=(1,2)).
5. **Model Training:** Built a pipeline with `LinearSVC` (Support Vector Machine).
6. **Hyperparameter Tuning:** Used `GridSearchCV` to find the best `C` parameter.
7. **Evaluation:** Evaluated using Accuracy, Precision, Recall, F1-Score, and 5-Fold Cross-Validation.
8. **Saving the Model:** Saved the trained model as `spam_svm_model.pkl`.

## 📈 Results
- **Accuracy:** ~98.4%
- **Precision:** ~98.3%
- **Recall:** ~89.3%
- **F1-Score:** ~93.6%
- **Mean Cross-Validation Accuracy:** ~97.95%

## 💻 How to Use
1. Clone this repository.
2. Ensure you have the required libraries installed (`pip install pandas scikit-learn nltk joblib`).
3. Open `spam_model.ipynb` in Jupyter or VS Code.
4. Load the saved model and use the prediction function:

```python
import joblib

# Load the saved model
model = joblib.load('spam_svm_model.pkl')

def predict_sms(text):
    # Note: Ensure you use the same 'clean_text' function from the notebook
    cleaned = clean_text(text) 
    prediction = model.predict([cleaned])[0]
    return "Spam 🚨" if prediction == 1 else "Ham ✅"

# Test
print(predict_sms("Congratulations! You have won a free iPhone. Click here now."))
print(predict_sms("Hi, are we meeting tomorrow at 3 PM?"))
