"""Collaborative Filtering Movie Recommendation System using co-occurrence."""

import os
from collections import defaultdict
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MOVIES_PATH = os.path.join(BASE_DIR, "dataset", "movies.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "dataset", "ratings.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "collaborativeout.txt")


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


def build_co_occurrence(ratings):
    """Count how often each pair of movies appears together in user ratings."""
    co_occur = defaultdict(int)
    for _, group in ratings.groupby("userId"):
        movie_ids = group["movieId"].values
        for i in range(len(movie_ids)):
            for j in range(len(movie_ids)):
                co_occur[(movie_ids[i], movie_ids[j])] += 1
    return co_occur


def score_movies(ratings, co_occur):
    """For each user, score unrated movies based on co-occurrence with rated movies.

    score(user, movie) = sum over rated movies m: rating(user, m) * co_occur(m, movie)
    """
    # Build per-user rating dict
    user_ratings = {}
    for _, row in ratings.iterrows():
        uid, mid, rating = int(row["userId"]), int(row["movieId"]), row["rating"]
        if uid not in user_ratings:
            user_ratings[uid] = {}
        user_ratings[uid][mid] = rating

    # Get all movie IDs
    all_movies = set(ratings["movieId"].unique())

    results = []
    for user_id, rated_movies in user_ratings.items():
        unrated = all_movies - set(rated_movies.keys())
        scores = {}
        for movie_id in unrated:
            score = 0.0
            for rated_mid, rating in rated_movies.items():
                count = co_occur.get((rated_mid, movie_id), 0)
                score += rating * count
            if score > 0:
                scores[movie_id] = score

        for movie_id, score in scores.items():
            results.append((user_id, movie_id, score))

    return results


def main():
    movies, ratings = load_data()

    print("Building co-occurrence matrix...")
    co_occur = build_co_occurrence(ratings)

    print("Scoring movies for each user...")
    scored = score_movies(ratings, co_occur)

    # Build results dataframe
    results_df = pd.DataFrame(scored, columns=["userId", "movieId", "score"])
    title_map = movies.set_index("movieId")["title"].to_dict()
    results_df["title"] = results_df["movieId"].map(title_map)
    # Sort by userId ascending, score descending
    results_df = results_df.sort_values(["userId", "score"], ascending=[True, False])

    # Write output (tab-separated to avoid issues with commas in titles)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for _, row in results_df.iterrows():
            f.write(f"{row['userId']}\t{row['movieId']}\t{row['title']}\t{row['score']}\n")

    print(f"Collaborative recommendations written to {OUTPUT_PATH}")
    print(f"Total recommendations: {len(results_df)}")


if __name__ == "__main__":
    main()
