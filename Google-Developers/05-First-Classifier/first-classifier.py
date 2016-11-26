# https://youtu.be/AoeEHqVSNOw

import math
from scipy.spatial import distance

class NearestNeightbors():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        smallest_distance = math.inf
        smallest_distance_index = None
        for i in range(1, len(self.x_train)):
            dist = distance.euclidean(row, self.x_train[i])
            if dist < smallest_distance:
                smallest_distance = dist
                smallest_distance_index = i
        return self.y_train[smallest_distance_index]
        

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/2)

# from sklearn import tree
# classifier = tree.DecisionTreeClassifier()

# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier()

classifier  = NearestNeightbors()

classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
