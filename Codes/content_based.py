"""Content-based Movie Recommendation System using TF-IDF on genre vectors."""

import os
import math
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MOVIES_PATH = os.path.join(BASE_DIR, "dataset", "movies.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "dataset", "ratings.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "contentout.txt")

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
    "Western", "(no genres listed)",
]


def load_data():
    movies = pd.read_csv(
        MOVIES_PATH, header=None, names=["movieId", "title", "genres"],
        dtype={"movieId": int, "title": str, "genres": str},
    )
    ratings = pd.read_csv(
        RATINGS_PATH, header=None, names=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": int, "movieId": int, "rating": float},
    )
    ratings = ratings.drop(columns=["timestamp"])
    return movies, ratings


def build_genre_matrix(movies):
    """Create a normalized genre vector for each movie (1/sqrt(num_genres))."""
    genre_data = {}
    for _, row in movies.iterrows():
        movie_id = row["movieId"]
        genres = row["genres"].split("|")
        norm = 1.0 / math.sqrt(len(genres))
        genre_data[movie_id] = {g: norm for g in genres}

    genre_df = pd.DataFrame.from_dict(genre_data, orient="index").fillna(0.0)
    # Ensure all genre columns exist
    for g in ALL_GENRES:
        if g not in genre_df.columns:
            genre_df[g] = 0.0
    genre_df = genre_df[ALL_GENRES]
    genre_df.index.name = "movieId"
    return genre_df


def compute_idf(movies):
    """Compute IDF for each genre: log(total_movies / genre_count)."""
    total_movies = len(movies)
    genre_counts = {}
    for genres_str in movies["genres"]:
        for g in genres_str.split("|"):
            genre_counts[g] = genre_counts.get(g, 0) + 1

    idf = {}
    for g in ALL_GENRES:
        count = genre_counts.get(g, 1)
        idf[g] = math.log(total_movies / count)
    return pd.Series(idf)


def build_user_profiles(ratings, genre_matrix):
    """Build user genre profiles: sum of (sentiment * genre_vector) for each rated movie."""
    # Convert ratings to binary sentiment
    sentiment = ratings.copy()
    sentiment["sentiment"] = np.where(sentiment["rating"] >= 2.5, 1, -1)

    # For each user, accumulate genre scores weighted by sentiment
    user_profiles = {}
    for user_id, group in sentiment.groupby("userId"):
        profile = np.zeros(len(ALL_GENRES))
        for _, row in group.iterrows():
            movie_id = row["movieId"]
            if movie_id in genre_matrix.index:
                profile += row["sentiment"] * genre_matrix.loc[movie_id].values
        user_profiles[user_id] = profile

    return pd.DataFrame.from_dict(user_profiles, orient="index", columns=ALL_GENRES)


def score_movies(user_profiles, genre_matrix, idf, ratings):
    """Score all unrated movies for each user using IDF-weighted dot product."""
    # Weight both profiles and movie vectors by IDF
    idf_values = idf[ALL_GENRES].values
    weighted_profiles = user_profiles.values * idf_values
    weighted_movies = genre_matrix.values * idf_values

    # Get set of rated movies per user
    rated = ratings.groupby("userId")["movieId"].apply(set).to_dict()

    results = []
    movie_ids = genre_matrix.index.values

    for i, user_id in enumerate(user_profiles.index):
        user_rated = rated.get(user_id, set())
        # Dot product of user profile with each movie's genre vector
        scores = weighted_profiles[i] @ weighted_movies.T
        for j, movie_id in enumerate(movie_ids):
            if movie_id not in user_rated:
                results.append((user_id, movie_id, scores[j]))

    return results


def main():
    movies, ratings = load_data()
    genre_matrix = build_genre_matrix(movies)
    idf = compute_idf(movies)
    user_profiles = build_user_profiles(ratings, genre_matrix)
    scored = score_movies(user_profiles, genre_matrix, idf, ratings)

    # Build results dataframe
    results_df = pd.DataFrame(scored, columns=["userId", "movieId", "score"])
    # Add movie titles
    title_map = movies.set_index("movieId")["title"].to_dict()
    results_df["title"] = results_df["movieId"].map(title_map)
    # Sort by userId ascending, score descending
    results_df = results_df.sort_values(["userId", "score"], ascending=[True, False])

    # Write output (tab-separated to avoid issues with commas in titles)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for _, row in results_df.iterrows():
            f.write(f"{row['userId']}\t{row['movieId']}\t{row['title']}\t{row['score']}\n")

    print(f"Content-based recommendations written to {OUTPUT_PATH}")
    print(f"Total recommendations: {len(results_df)}")


if __name__ == "__main__":
    main()
