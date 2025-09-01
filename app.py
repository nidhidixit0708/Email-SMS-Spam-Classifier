{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOKQyoxPdF6AnK48LJJ0R8l"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "CiQcH2aqu-4I"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "import pickle\n",
        "import string\n",
        "import sklearn\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem.porter import PorterStemmer\n",
        "\n",
        "# Ensure required NLTK resources are available\n",
        "nltk.download('punkt')\n",
        "nltk.download('stopwords')\n",
        "stop_words = set(stopwords.words(\"english\"))\n",
        "\n",
        "try:\n",
        "    nltk.data.find('corpora/stopwords')\n",
        "except LookupError:\n",
        "    print(\"Downloading the 'stopwords' resource...\")\n",
        "    try:\n",
        "        nltk.download('stopwords')\n",
        "        print(\"'stopwords' resource downloaded successfully.\")\n",
        "    except Exception as e:\n",
        "        print(f\"Error downloading 'stopwords': {e}\")\n",
        "\n",
        "ps = PorterStemmer()\n",
        "\n",
        "def transform_text(text):\n",
        "  text=text.lower()            #to lowercase\n",
        "  text=nltk.word_tokenize(text) # Corrected typo here\n",
        "  y=[]\n",
        "  for i in text:\n",
        "    if i.isalnum():\n",
        "      y.append(i)\n",
        "\n",
        "  text=y[:]\n",
        "  y.clear()\n",
        "\n",
        "  for i in text:\n",
        "    if i not in stopwords.words('english') and i not in string.punctuation:\n",
        "      y.append(i)\n",
        "\n",
        "  text=y[:]\n",
        "  y.clear()\n",
        "  for i in text:\n",
        "    y.append(ps.stem(i))\n",
        "\n",
        "  return \" \".join(y)\n",
        "\n",
        "# Load vectorizer and model\n",
        "tfidf = pickle.load(open('vectorizer.pkl', 'rb'))\n",
        "model = pickle.load(open('model.pkl', 'rb'))\n",
        "\n",
        "st.title(\"Email/SMS Spam Classifier\")\n",
        "\n",
        "input_sms = st.text_area(\"Enter the message\")\n",
        "\n",
        "if st.button(\"Predict\"):\n",
        "    # 1. Preprocess\n",
        "    transform_sms = transform_text(input_sms)\n",
        "    # 2. Vectorize\n",
        "    vector_input = tfidf.transform([transform_sms])\n",
        "    # 3. Predict\n",
        "    result = model.predict(vector_input)[0]\n",
        "    # 4. Display\n",
        "    if result == 1:\n",
        "        st.header(\"Spam\")\n",
        "    else:\n",
        "        st.header(\"Not Spam\")\n"
      ]
    }
  ]
}