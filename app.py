from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import pickle
import os
import zipfile

# Initialize Flask app
app = Flask(__name__, template_folder='templatessss')

zip_file_path = 'notebooks/notebooks.zip'
extract_path = 'notebooks/extracted/'

# Extract the zip file if it hasn't been extracted yet
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Define the file paths for the model and the dataset
model_path = os.path.join(extract_path, 'KNN_model.pkl')
movies_df_path = os.path.join(extract_path, 'movies_df.pkl')

# Load the model and dataset
model = pickle.load(open(model_path, 'rb'))
movies_df = pickle.load(open(movies_df_path, 'rb'))

# Create a sparse matrix for the KNN model
movies_df_matrix = csr_matrix(movies_df.values)

# Build and train the KNN model
model.fit(movies_df_matrix)

@app.route('/')
def home():
    # Get a list of all movie titles
    movie_titles = list(movies_df.index)
    return render_template('home.html', movies=movie_titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form.get('movie_name')
    
    if movie_name not in movies_df.index:
        return render_template('result.html', error='Movie not found.', recommendations=[])

    # Get the index of the movie
    movie_idx = movies_df.index.get_loc(movie_name)

    # Get recommendations
    distances, indices = model.kneighbors(movies_df.iloc[movie_idx, :].values.reshape(1, -1), n_neighbors=6)

    recommendations = []
    for i in range(0, len(distances.flatten())):
        if i != 0:  # Skip the first one as it's the same movie
            recommendations.append(f'{i}: {movies_df.index[indices.flatten()[i]]}, with distance of {distances.flatten()[i]:.4f}')

    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)




#recommendations = []
#for i in range(1, len(distances.flatten())):  # Start at 1 to skip the first one
 #   recommendations.append({
  ###    'distance': distances.flatten()[i]
    #})