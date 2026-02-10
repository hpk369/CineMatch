"""Collaborative Filtering — Phase 1: Build co-occurrence matrix from ratings.

Hadoop MapReduce implementation via mrjob.
Usage: python cooccurrence.py -r hadoop dataset/ratings.csv -o output/cooccur/
"""

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol


class CoOccurrenceMRJob(MRJob):

    OUTPUT_PROTOCOL = RawValueProtocol

    def mapper(self, _, line):
        """Parse ratings.csv → yield (userId, movieId)"""
        parts = line.strip().split(",")
        if len(parts) < 3:
            return
        try:
            user_id = int(parts[0])
            movie_id = int(parts[1])
        except (ValueError, IndexError):
            return
        yield user_id, movie_id

    def reducer_pairs(self, user_id, movie_ids):
        """Collect all movies for a user, emit all pairs."""
        movies = list(movie_ids)
        for m1 in movies:
            for m2 in movies:
                yield (m1, m2), 1

    def reducer_count(self, pair, counts):
        """Sum co-occurrence counts for each movie pair."""
        total = sum(counts)
        yield None, "%d\t%d\t%d" % (pair[0], pair[1], total)

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer_pairs),
            MRStep(reducer=self.reducer_count),
        ]


if __name__ == "__main__":
    CoOccurrenceMRJob.run()
