"""Generate compact JSON data files for the website from recommendation outputs."""

import os
import json
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
WEBSITE_DIR = os.path.join(BASE_DIR, "docs")
MOVIES_PATH = os.path.join(BASE_DIR, "dataset", "movies.csv")

TOP_N = 50  # top recommendations per user per method


def load_movies_with_genres():
    """Load movies and build a movieId -> genres mapping."""
    movie_genres = {}
    with open(MOVIES_PATH, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            movie_id = parts[0]
            genres = parts[-1].split("|")
            movie_genres[movie_id] = genres
    return movie_genres


def load_recommendations(filename):
    """Load tab-separated recommendation file into {userId: [(movieId, title, score)]}."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    recs = defaultdict(list)
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            user_id = parts[0]
            movie_id = parts[1]
            title = parts[2]
            score = parts[3]
            if len(recs[user_id]) < TOP_N:
                recs[user_id].append({
                    "movieId": movie_id,
                    "title": title,
                    "score": round(float(score), 2),
                })
    return dict(recs)


def main():
    print("Loading movie genres...")
    movie_genres = load_movies_with_genres()

    print("Loading content-based recommendations...")
    content = load_recommendations("contentout.txt")

    print("Loading collaborative recommendations...")
    collab = load_recommendations("collaborativeout.txt")

    print("Loading hybrid recommendations...")
    hybrid = load_recommendations("hybridout.txt")

    # Add genre info to each recommendation
    for recs in [content, collab, hybrid]:
        for user_id in recs:
            for rec in recs[user_id]:
                rec["genres"] = movie_genres.get(rec["movieId"], [])

    # Get all user IDs
    all_users = sorted(set(content.keys()) | set(collab.keys()) | set(hybrid.keys()), key=int)

    data = {
        "users": all_users,
        "genres": [
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ],
        "content": content,
        "collaborative": collab,
        "hybrid": hybrid,
    }

    os.makedirs(WEBSITE_DIR, exist_ok=True)
    output_path = os.path.join(WEBSITE_DIR, "data.json")
    with open(output_path, "w") as f:
        json.dump(data, f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Website data written to {output_path} ({size_mb:.1f} MB)")
    print(f"Users: {len(all_users)}, Top {TOP_N} recommendations per user per method")


if __name__ == "__main__":
    main()
