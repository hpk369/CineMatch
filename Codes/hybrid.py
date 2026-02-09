"""Hybrid Movie Recommendation System — combines content-based and collaborative scores."""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CONTENT_PATH = os.path.join(BASE_DIR, "output", "contentout.txt")
COLLAB_PATH = os.path.join(BASE_DIR, "output", "collaborativeout.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "hybridout.txt")


def load_recommendations(path):
    return pd.read_csv(
        path, header=None, names=["userId", "movieId", "title", "score"],
        sep="\t",
        dtype={"userId": int, "movieId": int, "title": str, "score": float},
    )


def main():
    content = load_recommendations(CONTENT_PATH)
    collab = load_recommendations(COLLAB_PATH)

    # Merge on userId and movieId
    merged = pd.merge(
        content, collab,
        on=["userId", "movieId", "title"],
        suffixes=("_content", "_collab"),
    )

    # Hybrid score = content_score * collaborative_score
    merged["score"] = merged["score_content"] * merged["score_collab"]

    # Sort by userId ascending, score descending
    merged = merged.sort_values(["userId", "score"], ascending=[True, False])

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for _, row in merged.iterrows():
            f.write(f"{row['userId']}\t{row['movieId']}\t{row['title']}\t{row['score']}\n")

    print(f"Hybrid recommendations written to {OUTPUT_PATH}")
    print(f"Total recommendations: {len(merged)}")


if __name__ == "__main__":
    main()
