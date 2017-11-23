import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Loading columns
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

#Reading from URL with pandas
dataset = pandas.read_csv(url, names = names)

# Show shape of dataset
print('Data set size: {}'.format(dataset.shape))

# Eyeball first 10 records
print('----------FIRST 10 RECORDS----------')
print(dataset.head(10))

# Show statistical description
print('----------SUMMARY----------')
print(dataset.describe())

# Grouping by class
groups = dataset.groupby('class')
print('----------COUNT BY CLASS----------')
print(groups.size())

# Visualizing box plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# Visualizing in hist plot
dataset.hist()
# plt.show()

# Scatter plot matrix
scatter_matrix(dataset)
# plt.show()

# Preparing training and test data
values = dataset.values
X = values[:,0:4]
Y = values[:, 4]
test_proportion = 0.2
seed = 7
X_training, X_test, Y_training, Y_test = model_selection.train_test_split(X, Y, test_size = test_proportion, random_state = seed)

# Defining performance metric to evaluate models
scoring = 'accuracy'

# Adding models to test
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Traning models
results = []
names = []
print('----------CROSS VALIDATION----------')
for name, model in models:
  # Config to Split data set in K consecutive folds
	kfold = model_selection.KFold(n_splits=10, random_state=seed)

  # Makes Cross validation proccess given the parameters: Model, Features, Labels, Cross Val splitting strategy and Performance metric
	cv_results = model_selection.cross_val_score(model, X_training, Y_training, cv=kfold, scoring=scoring)

	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
# plt.show()

# Then we evaluate which one is the best algoritm.
# In this case, SVM has the best acuracy, so, we will predict with this model
bestModel = SVC()
bestModel.fit(X_training, Y_training)
predictions = bestModel.predict(X_test)

# Then, we will analyze the metrics
print('----------BEST MODEL METRICS WITH NEW DATA----------')
accuracy = accuracy_score(Y_test, predictions)
print('Accuracy: {}'.format(accuracy))
matrix = confusion_matrix(Y_test, predictions)
print('Confusion matrix: ')
print(matrix)
report = classification_report(Y_test, predictions)
print('Classification report: ')
print(report)