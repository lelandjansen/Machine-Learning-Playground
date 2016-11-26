# https://youtu.be/cKxRvEZd3Mw

from sklearn import tree

# bumpy -> 0
# smooth -> 1
features = [
    # mass (g), texture
    [140, 1],
    [130, 1],
    [150, 0],
    [170, 0]
]

# apple -> 0
# orange -> 1
labels = [
    0,
    0,
    1,
    1
]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels) # fit -> find features in data

print('orange' if classifier.predict([[150, 0]]) else 'apple')
