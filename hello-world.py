from sklearn import tree

# We will define here the qualities to use them, but take into account that
# they have to be integers!
bumpy = 1
smooth = 0

apple = 1
orange = 0


# Define the list of instances to feed the classifier...
features = [[140, smooth], [130, smooth], [150, bumpy], [170, bumpy]]

# ... and the labels associated to each of the instances.
labels = [apple, apple, orange, orange]

# Define the classifier itself
clf = tree.DecisionTreeClassifier()

# Feed the classifier with the examples
clf = clf.fit(features, labels)

# Let's see what the classifier predicts for this example, should be an orange
if (clf.predict([[160, bumpy]]) == apple): 
    print "Apple"
else:
    print "Orange"
