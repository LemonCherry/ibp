import re

delimiter = re.compile(r'[\s,]+')


class InferenceBase:
    def __init__(self, cutoff=10, enable_cluster=True):
        self.cutoff = cutoff
        self.enable_cluster = enable_cluster

    def fit_from_file(self, fn):
        arr = map(lambda x: map(int, delimiter.split(x.strip())), open(fn))
        cats = map(lambda x: x[0], arr)
        xs = map(lambda x: x[1], arr)
        ns = map(lambda x: x[2], arr)
        self.fit(ns, xs, cats)

    def fit(self, ns, xs, cats):
        pass

    def __call__(self, n, x, cat=None):
        pass


class PER(InferenceBase):
    def __call__(self, n, x, cat=None):
        return float(x) / n

    def __init__(self, cutoff=10, enable_cluster=True):
        InferenceBase.__init__(self, cutoff, enable_cluster)
        self.__name__ = 'PER'


class Review:
    def __init__(self, feature_vector, pos, neg):
        self.x = feature_vector
        self.pos = int(pos)
        self.neg = int(neg)
        self.total = self.pos + self.neg
        self.h = self.pos / float(self.total)
        self.y_ibp = {1: True, 2: True, 4: True, 8: True, 16: True}
        self.categories = {1: 0, 2: -1, 4: -1, 8: -1, 16: -1}
        self.is_train = {1: True, 2: True, 4: True, 8: True, 16: True}


class Dataset:
    def __init__(self, reviews):
        """
        :type reviews: list[Review]
        :param reviews:
        """
        self.reviews = reviews

