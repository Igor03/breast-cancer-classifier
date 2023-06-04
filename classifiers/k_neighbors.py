from sklearn.neighbors import KNeighborsClassifier


class KNeighbors:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train(self) -> KNeighborsClassifier:

        classifier = KNeighborsClassifier(
            n_neighbors=3,
            algorithm='auto',
            leaf_size=30,
            p=2)

        classifier.fit(self.x, self.y)

        return classifier
