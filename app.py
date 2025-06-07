from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import requests
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model, feature columns, and data once at startup
model = joblib.load('random_forest_movie_rating_model.joblib')
feature_cols = joblib.load('feature_columns.joblib')

# Load merged data including imdbId_y
df = pd.read_csv('merged_with_imdb.csv')

# Rename imdbId_y to imdbId for easier reference
df.rename(columns={'imdbId_y': 'imdbId'}, inplace=True)

OMDB_API_KEY = os.getenv('API_KEY')  # Replace with your valid OMDb API key

def fetch_movie_details(imdb_id):
    url = "http://www.omdbapi.com/"
    params = {
        'apikey': OMDB_API_KEY,
        'i': imdb_id,
        'plot': 'full',
        'r': 'json'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        # Return the full OMDb JSON response
        return data
    except requests.RequestException:
        # In case of any error, return empty response structure
        return {}

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    genre = data.get('genre')
    min_rating = float(data.get('min_rating', 0))
    min_votes = int(data.get('min_votes', 0))

    if not genre or genre not in df.columns:
        return jsonify({'error': 'Invalid or missing genre'}), 400

    # Filter movies based on user criteria
    filtered_df = df[
        (df[genre] == 1) &
        (df['rating_count'] >= min_votes) &
        (df['avg_rating'] >= min_rating)
    ].copy()

    if filtered_df.empty:
        return jsonify({'recommendations': []})

    # Prepare features for prediction
    X = filtered_df[feature_cols]

    # Predict ratings
    filtered_df['predicted_rating'] = model.predict(X)

    # Sort by predicted rating descending and take top 10
    recommendations = filtered_df.sort_values('predicted_rating', ascending=False).head(10)

    results = []
    for _, row in recommendations.iterrows():
        imdb_id_int = int(row['imdbId'])
        imdb_id = f'tt{imdb_id_int:07d}'
        omdb_response = fetch_movie_details(imdb_id)

        result = {
            'movieId': row['movieId'],
            'imdbId': imdb_id,
            'title': row['title'],
            'genres': row['genres'],
            'avg_rating': row['avg_rating'],
            'rating_count': row['rating_count'],
            'predicted_rating': row['predicted_rating'],
            'omdb_response': omdb_response  # full OMDb data here
        }
        results.append(result)

    return jsonify({'recommendations': results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
