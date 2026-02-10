"""Hybrid Movie Recommendation System — combines content-based and collaborative scores.

Hadoop MapReduce implementation via mrjob.
Usage: python hybrid.py -r hadoop --content output/contentout.txt --collab output/collaborativeout.txt dataset/ratings.csv -o output/hybrid/
"""

from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import RawValueProtocol

SORT_CONSTANT = 1e12


class HybridMRJob(MRJob):

    OUTPUT_PROTOCOL = RawValueProtocol

    def configure_args(self):
        super().configure_args()
        self.add_file_arg("--content", help="Path to content-based output file")
        self.add_file_arg("--collab", help="Path to collaborative output file")

    def load_recommendations(self, filepath):
        """Load a tab-separated recommendation file into a dict."""
        recs = {}
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                try:
                    user_id = int(parts[0])
                    movie_id = int(parts[1])
                    title = parts[2]
                    score = float(parts[3])
                    recs[(user_id, movie_id)] = (title, score)
                except (ValueError, IndexError):
                    continue
        return recs

    def mapper_init(self):
        """Load both recommendation files as side files."""
        self.content = self.load_recommendations(self.options.content)
        self.collab = self.load_recommendations(self.options.collab)

    def mapper(self, _, line):
        """No-op — main input is ignored, all work done in mapper_final."""
        pass

    def mapper_final(self):
        """Merge content and collaborative scores by multiplication."""
        for key in self.content:
            if key in self.collab:
                user_id, movie_id = key
                title = self.content[key][0]
                hybrid_score = self.content[key][1] * self.collab[key][1]
                sort_key = "%06d_%020.6f" % (user_id, SORT_CONSTANT - hybrid_score)
                yield sort_key, "%d\t%d\t%s\t%.6f" % (
                    user_id, movie_id, title, hybrid_score
                )

    def reducer_sort(self, _, values):
        """Deduplicate and pass-through — mapper_final may run on multiple mappers."""
        seen = False
        for val in values:
            if not seen:
                yield None, val
                seen = True

    def steps(self):
        return [
            MRStep(
                mapper_init=self.mapper_init,
                mapper=self.mapper,
                mapper_final=self.mapper_final,
                reducer=self.reducer_sort,
            ),
        ]


if __name__ == "__main__":
    HybridMRJob.run()
