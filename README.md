# CineMatch

A hybrid movie recommendation engine built on **Hadoop MapReduce** using **Python (mrjob)**, implementing three recommendation approaches:

- **Content-Based Filtering** — TF-IDF weighted genre similarity
- **Collaborative Filtering** — user co-occurrence patterns
- **Hybrid** — combines both scores by multiplication

## Architecture

```
ratings.csv --> Content-Based (mrjob, 2 MRSteps) --> contentout.txt --+
                                                                      |
ratings.csv --> Co-occurrence (mrjob, 2 MRSteps) --> cooccur.txt      +--> Hybrid (mrjob) --> hybridout.txt
                     |                                                |
                     v                                                |
               Scoring (mrjob, 2 MRSteps) --> collaborativeout.txt ---+
```

## Requirements

- Python 3.7+
- mrjob
- Hadoop cluster (HDFS + YARN)

```bash
pip install mrjob
```

## Dataset

Uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/) in `dataset/`:
- `movies.csv` — 9,125 movies with genres
- `ratings.csv` — 100,004 ratings from 671 users

## Running

### Full Pipeline (Hadoop)

```bash
bash Codes/run_all.sh
```

### Individual Jobs

```bash
# Content-based
python Codes/content_based.py -r hadoop --movies dataset/movies.csv dataset/ratings.csv -o output/content/

# Collaborative — Phase 1: Co-occurrence
python Codes/cooccurrence.py -r hadoop dataset/ratings.csv -o output/cooccur/

# Collaborative — Phase 2: Scoring
python Codes/collaborative.py -r hadoop \
    --movies dataset/movies.csv \
    --cooccurrence output/cooccur/part-00000 \
    dataset/ratings.csv -o output/collab/

# Hybrid
python Codes/hybrid.py -r hadoop \
    --content output/content/part-00000 \
    --collab output/collab/part-00000 \
    dataset/ratings.csv -o output/hybrid/
```

### Local Testing (no Hadoop required)

Replace `-r hadoop` with `-r local` to run on your machine:

```bash
python Codes/content_based.py -r local --movies dataset/movies.csv dataset/ratings.csv -o output/content/
```

## Output

Results are written to `output/`:
- `contentout.txt` — content-based recommendations
- `collaborativeout.txt` — collaborative filtering recommendations
- `hybridout.txt` — hybrid recommendations

Format (tab-separated): `userId\tmovieId\tmovieTitle\tscore` (sorted by userId ascending, score descending)

## Website

An interactive cinema-themed web interface for exploring recommendations:

```bash
python Codes/generate_web_data.py        # Generate JSON from output files
cd website && python -m http.server      # Serve locally
```

## Tech Stack

- **Processing:** Python, mrjob, Hadoop MapReduce (Streaming)
- **Data:** MovieLens 100K (CSV)
- **Algorithms:** TF-IDF, Co-occurrence Matrix, Score Fusion
- **Web:** HTML5, CSS3, JavaScript
