from sklearn.ensemble import AdaBoostClassifier


class AdaBoost:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train(self) -> AdaBoostClassifier:

        classifier = AdaBoostClassifier(
            n_estimators=50,
            random_state=42)

        classifier.fit(self.x, self.y)

        return classifier
