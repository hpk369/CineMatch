"""Collaborative Filtering — Phase 2: Score unrated movies using co-occurrence.

Hadoop MapReduce implementation via mrjob.
Usage: python collaborative.py -r hadoop --movies dataset/movies.csv --cooccurrence output/cooccur/part-00000 dataset/ratings.csv -o output/collab/
"""

from collections import defaultdict
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol

SORT_CONSTANT = 1e12


def parse_movie_line(line):
    """Parse a movies.csv line handling commas in titles."""
    line = line.strip()
    if not line:
        return None
    last_comma = line.rfind(",")
    genres_str = line[last_comma + 1:]
    rest = line[:last_comma]
    first_comma = rest.index(",")
    movie_id = int(rest[:first_comma])
    title = rest[first_comma + 1:]
    return movie_id, title


class CollaborativeScoringMRJob(MRJob):

    OUTPUT_PROTOCOL = RawValueProtocol

    def configure_args(self):
        super().configure_args()
        self.add_file_arg("--movies", help="Path to movies.csv")
        self.add_file_arg("--cooccurrence", help="Path to co-occurrence output file")

    def load_side_files(self):
        """Load movies.csv for titles and co-occurrence file for scoring."""
        # Load movie titles
        self.title_map = {}
        self.all_movies = set()
        with open(self.options.movies, "r") as f:
            for line in f:
                parsed = parse_movie_line(line)
                if parsed is None:
                    continue
                movie_id, title = parsed
                self.title_map[movie_id] = title
                self.all_movies.add(movie_id)

        # Load co-occurrence counts
        self.co_occur = defaultdict(int)
        with open(self.options.cooccurrence, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    m1 = int(parts[0])
                    m2 = int(parts[1])
                    count = int(parts[2])
                    self.co_occur[(m1, m2)] = count
                except (ValueError, IndexError):
                    continue

    def mapper_init(self):
        self.load_side_files()

    def mapper(self, _, line):
        """Parse ratings.csv → yield (userId, (movieId, rating))"""
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
        self.load_side_files()

    def reducer(self, user_id, values):
        """Score all unrated movies based on co-occurrence with rated movies."""
        rated = {}
        for movie_id, rating in values:
            rated[movie_id] = rating

        for candidate in self.all_movies:
            if candidate in rated:
                continue
            score = 0.0
            for rated_mid, rating in rated.items():
                count = self.co_occur.get((rated_mid, candidate), 0)
                score += rating * count
            if score > 0:
                title = self.title_map.get(candidate, "Unknown")
                sort_key = "%06d_%020.6f" % (user_id, SORT_CONSTANT - score)
                yield sort_key, "%d\t%d\t%s\t%.6f" % (user_id, candidate, title, score)

    # ── Sort pass-through ──

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
    CollaborativeScoringMRJob.run()
