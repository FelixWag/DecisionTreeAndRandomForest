# DecisionTreeAndRandomForest
Python3 implementation of Decision Tree and Random Forest (gini-index as the attribute selection measure) for classification

In the Data folder there are some test and training data files. In both implementations there are hyperparamater tuned for every data set in the data folder.
If new data sets are included, the hyperparamters have to be added to the implementations.

Usage (DecisionTree):
python3 DecisionTree.py train_file test_file

Usage (RandomForst):
python3 RandomForest.py train_file test_file

Output: 
k*k (k = count of labels) matrix representing the confusion matrix of the classifier on testing data 

Example usage (with data from the data folder):
python3 RandomForest balance.scale.train balance.scale.test
