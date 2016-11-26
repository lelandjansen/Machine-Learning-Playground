# https://youtu.be/tNa99PG8hR8

import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
test_index = [0, 50, 100]

# training
train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis = 0)

# test
test_target = iris.target[test_index]
test_data = iris.data[test_index]

from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_target)

print(test_target)
print(classifier.predict(test_data))
