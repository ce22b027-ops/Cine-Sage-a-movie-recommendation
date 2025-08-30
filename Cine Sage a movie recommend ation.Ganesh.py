import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Sample Movie Dataset
# --------------------------
movies_data = {
    'title': [
        'The Matrix',
        'Inception',
        'Interstellar',
        'The Dark Knight',
        'The Godfather',
        'The Shawshank Redemption',
        'Pulp Fiction',
        'Fight Club',
        'Forrest Gump',
        'The Lord of the Rings'
    ],
    'genre': [
        'Sci-Fi Action',
        'Sci-Fi Thriller',
        'Sci-Fi Drama',
        'Action Crime Drama',
        'Crime Drama',
        'Drama',
        'Crime Drama',
        'Drama Thriller',
        'Romance Drama',
        'Fantasy Adventure'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(movies_data)

# --------------------------
# Step 1: Vectorize Genres
# --------------------------
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df['genre'])

# --------------------------
# Step 2: Compute Similarity
# --------------------------
similarity = cosine_similarity(genre_matrix)

# --------------------------
# Recommendation Function
# --------------------------
def recommend_movies(title, num_recommendations=5):
    if title not in df['title'].values:
        print(f"Sorry, '{title}' not found in Cine Sage movie list.")
        return
    
    idx = df[df['title'] == title].index[0]
    similarity_scores = list(enumerate(similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print(f"\n🎬 Because you liked **{title}**, Cine Sage recommends:")
    count = 0
    for i, score in similarity_scores[1:]:
        print(f"- {df.iloc[i]['title']} ({df.iloc[i]['genre']})")
        count += 1
        if count == num_recommendations:
            break

# --------------------------
# Run Cine Sage
# --------------------------
print("🎥 Welcome to Cine Sage — Your Movie Recommendation Buddy!")
print("Available Movies:")
print(df['title'].to_string(index=False))

while True:
    movie = input("\nEnter a movie title (or type 'exit' to quit): ").strip()
    if movie.lower() == 'exit':
        print("Goodbye from Cine Sage! 🍿")
        break
    recommend_movies(movie)
