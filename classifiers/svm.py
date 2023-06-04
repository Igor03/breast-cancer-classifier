from sklearn.svm import SVC


class SVM:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train(self) -> SVC:

        classifier = SVC(
            kernel="rbf",
            C=2.7,
            coef0=0.001,
            random_state=42)

        classifier.fit(self.x, self.y)

        return classifier
