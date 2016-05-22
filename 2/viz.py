import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# Iris is an example dataset provided by scikit-learn
iris = load_iris()

# Choose 3 examples as training data. The set is ordered, so we can use the
# first one of each type of flower.
test_idx = [0, 50, 100]

# Get the training data by removing the testing data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Get the testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Train the classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# Test the classifier
results = clf.predict(test_data)
for (result, real) in zip(results, test_target):
    if (result == real):
        print "Example {n} correctly classified".format(n=real)
    else:
        print "Example {n} not correctly classified".format(n=real)

# The classifier should correctly classify every result. To see the decission
# tree, we can use graphviz to disply it.
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())

# To make it work, you need to have graphviz installed in your computer
try:
    graph.write_pdf("iris-tree.pdf")
except pydot.InvocationException:
    print "You need to have graphviz installed in you system"
