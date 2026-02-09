# Movie-Recommendation-System

A movie recommendation system implementing three approaches:
- **Content-Based Filtering** — recommends movies based on genre similarity using TF-IDF
- **Collaborative Filtering** — recommends movies based on co-occurrence patterns across users
- **Hybrid** — combines both scores by multiplication

## Requirements

- Python 3.7+ 
- pandas
- numpy

Install dependencies:
```bash
pip install pandas numpy
```

## Dataset

Uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/) located in `dataset/`:
- `movies.csv` — 9,125 movies with genres
- `ratings.csv` — 100,004 ratings from 671 users

## Running

Run each recommender from the project root:

```bash
# Content-based recommendations
python Codes/content_based.py

# Collaborative filtering recommendations
python Codes/collaborative.py

# Hybrid (requires content and collaborative output first)
python Codes/hybrid.py
```

## Output

Results are written to the `output/` directory:
- `contentout.txt` — content-based recommendations
- `collaborativeout.txt` — collaborative filtering recommendations
- `hybridout.txt` — hybrid recommendations

Output format (tab-separated): `userId\tmovieId\tmovieTitle\tscore` (sorted by userId ascending, score descending)
