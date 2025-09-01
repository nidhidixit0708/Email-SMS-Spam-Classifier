import streamlit as st
import pickle
import string
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading the 'stopwords' resource...")
    try:
        nltk.download('stopwords')
        print("'stopwords' resource downloaded successfully.")
    except Exception as e:
        print(f"Error downloading 'stopwords': {e}")

ps = PorterStemmer()

def transform_text(text):
  text=text.lower()            #to lowercase
  text=nltk.word_tokenize(text) # Corrected typo here
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. Preprocess
    transform_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
