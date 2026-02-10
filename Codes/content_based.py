"""Content-based Movie Recommendation System using TF-IDF on genre vectors.

Hadoop MapReduce implementation via mrjob.
Usage: python content_based.py -r hadoop --movies dataset/movies.csv dataset/ratings.csv -o output/content/
"""

import math
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
    "Western", "(no genres listed)",
]
NUM_GENRES = len(ALL_GENRES)
GENRE_INDEX = {g: i for i, g in enumerate(ALL_GENRES)}
SORT_CONSTANT = 1e12


def parse_movie_line(line):
    """Parse a movies.csv line handling commas in titles.

    Format: movieId,title,genres (genres are pipe-separated, never contain commas)
    """
    line = line.strip()
    if not line:
        return None
    last_comma = line.rfind(",")
    genres_str = line[last_comma + 1:]
    rest = line[:last_comma]
    first_comma = rest.index(",")
    movie_id = int(rest[:first_comma])
    title = rest[first_comma + 1:]
    genres = genres_str.split("|")
    return movie_id, title, genres


class ContentBasedMRJob(MRJob):

    OUTPUT_PROTOCOL = RawValueProtocol

    def configure_args(self):
        super().configure_args()
        self.add_file_arg("--movies", help="Path to movies.csv")

    def load_movies(self):
        """Load movies.csv into genre vectors, IDF scores, and title map."""
        self.genre_vectors = {}
        self.title_map = {}
        genre_counts = [0] * NUM_GENRES
        total_movies = 0

        with open(self.options.movies, "r") as f:
            for line in f:
                parsed = parse_movie_line(line)
                if parsed is None:
                    continue
                movie_id, title, genres = parsed
                total_movies += 1
                self.title_map[movie_id] = title

                # Normalized genre vector: 1/sqrt(num_genres) for each genre present
                vec = [0.0] * NUM_GENRES
                norm = 1.0 / math.sqrt(len(genres))
                for g in genres:
                    idx = GENRE_INDEX.get(g)
                    if idx is not None:
                        vec[idx] = norm
                        genre_counts[idx] += 1
                self.genre_vectors[movie_id] = vec

        # Compute IDF: log(total_movies / genre_count)
        self.idf = [0.0] * NUM_GENRES
        for i in range(NUM_GENRES):
            count = genre_counts[i] if genre_counts[i] > 0 else 1
            self.idf[i] = math.log(total_movies / count)

    # ── Step 1: Build user profiles and score all unrated movies ──

    def mapper_init(self):
        self.load_movies()

    def mapper(self, _, line):
        """Parse ratings.csv: userId,movieId,rating,timestamp"""
        parts = line.strip().split(",")
        if len(parts) < 3:
            return
        try:
            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = float(parts[2])
        except (ValueError, IndexError):
            return
        yield user_id, (movie_id, rating)

    def reducer_init(self):
        self.load_movies()

    def reducer(self, user_id, values):
        """Build user profile from ratings, then score all unrated movies."""
        ratings_list = list(values)
        rated_set = set()

        # Build user genre profile
        profile = [0.0] * NUM_GENRES
        for movie_id, rating in ratings_list:
            rated_set.add(movie_id)
            sentiment = 1 if rating >= 2.5 else -1
            vec = self.genre_vectors.get(movie_id)
            if vec:
                for i in range(NUM_GENRES):
                    profile[i] += sentiment * vec[i]

        # Score every unrated movie: IDF-weighted dot product
        for movie_id, vec in self.genre_vectors.items():
            if movie_id in rated_set:
                continue
            score = sum(
                self.idf[i] * profile[i] * vec[i] for i in range(NUM_GENRES)
            )
            title = self.title_map.get(movie_id, "Unknown")
            sort_key = "%06d_%020.6f" % (user_id, SORT_CONSTANT - score)
            yield sort_key, "%d\t%d\t%s\t%.6f" % (user_id, movie_id, title, score)

    # ── Step 2: Sort pass-through ──

    def reducer_sort(self, _, values):
        for val in values:
            yield None, val

    def steps(self):
        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper,
                reducer_init=self.reducer_init,
                reducer=self.reducer,
            ),
            MRStep(reducer=self.reducer_sort),
        ]


if __name__ == "__main__":
    ContentBasedMRJob.run()
