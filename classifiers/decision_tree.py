from sklearn.tree import DecisionTreeClassifier


class DecisionTree:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train(self) -> DecisionTreeClassifier:

        classifier = DecisionTreeClassifier(
            criterion='gini',
            min_samples_split=5,
            min_samples_leaf=5,
            max_depth=12,
            random_state=42)

        classifier.fit(self.x, self.y)

        return classifier
