from flask import Flask, request, render_template, jsonify
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the dataset (Replace with the actual path to your dataset)
data = pd.read_csv("TripAdvisor_RestauarantRecommendation.csv")

# Preprocess the text data (same as in your original script)
def preprocess_text(text):
    if isinstance(text, str):  # Check if the value is a string
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stopwords.words("english")]
        return " ".join(tokens)
    return ""

data["Processed_Type"] = data["Type"].apply(preprocess_text)
data["Processed_Location"] = data["Location"].apply(preprocess_text)

# TF-IDF vectorization (same as in your original script)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data["Processed_Type"] + " " + data["Processed_Location"])

# Serve the front-end HTML
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint to get restaurant recommendations
@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        # Get user preferences from the form
        user_type = request.json.get("restaurant-type")
        user_location = request.json.get("restaurant-location")
        processed_user_input = preprocess_text(user_type + " " + user_location)
        input_vector = tfidf_vectorizer.transform([processed_user_input])

        # Calculate cosine similarity (same as in your original script)
        similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
        top_indices = similarity_scores.argsort()[-10:][::-1]

        # Prepare recommended restaurants data
        recommended_restaurants = []
        for idx in top_indices:
            recommended_restaurant = data.loc[idx]
            recommended_restaurants.append({
                "Name": recommended_restaurant["Name"],
                "Street Address": recommended_restaurant["Street Address"],
                "Location": recommended_restaurant["Location"],
                # ... (Include other relevant fields)
            })

        return jsonify({"recommendations": recommended_restaurants})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)