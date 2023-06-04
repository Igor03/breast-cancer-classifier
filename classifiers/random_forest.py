from sklearn.ensemble import RandomForestClassifier


class RandomForest:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train(self) -> RandomForestClassifier:

        classifier = RandomForestClassifier(
            n_estimators=250,
            criterion='gini',
            max_depth=12,
            random_state=42)

        classifier.fit(self.x, self.y)

        return classifier
