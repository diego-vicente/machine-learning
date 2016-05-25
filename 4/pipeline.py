from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Import iris dataset
iris = datasets.load_iris()

# We will refer to the data and the target as X and y, to mimic the funcion
# definition: f(X) = y
X = iris.data
y = iris.target

# Split the set in half, to train and to test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# Create and train the classifiers
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

k_neighbours = KNeighborsClassifier()
k_neighbours.fit(X_train, y_train)

# Generate the results of the test set
score_dt = accuracy_score(y_test, decision_tree.predict(X_test))
score_kn = accuracy_score(y_test, k_neighbours.predict(X_test))

# Print the results of the classifier
print "The score of the Decission Tree is {}".format(score_dt)
print "The score of the KNeighbors is {}".format(score_kn)
