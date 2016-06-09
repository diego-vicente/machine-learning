import random
from scipy.spatial import distance

def euclidean(a, b):
    return distance.euclidean(a,b)

class RandomClassifier():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def closest(self, row):
        """Computes the closes neighbor to a given instance"""
        best_dist = euclidean(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euclidean(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

rnd_classifier = RandomClassifier()
knn_classifier = ScrappyKNN()

rnd_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)

rnd_predictions = rnd_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print "The random classifier scores {score}".format(
    score=accuracy_score(y_test, rnd_predictions))
print "The 1-NN classifier scores {score}".format(
    score=accuracy_score(y_test, knn_predictions))
