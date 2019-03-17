from collections import Counter
import DecisionTree
import sys

def main():
    test_file = "Data/" + sys.argv[2]
    trainings_file = "Data/" + sys.argv[1]

    max_depth = 100
    number_trees = 40
    number_attr = 3
    percent_sample = 40


    # Set the  Hyper parameters
    if "balance.scale" in trainings_file:
        max_depth = 100
        number_trees = 100
        number_attr = 2
        percent_sample = 20
    elif "nursery" in trainings_file:
        max_depth = 100
        number_trees = 100
        number_attr = 3
        percent_sample = 30
    elif "led" in trainings_file:
        max_depth = 100
        number_trees = 60
        number_attr = 2
        percent_sample = 40
    elif "synthetic.social" in trainings_file:
        max_depth = 100
        number_trees = 100
        number_attr = 11
        percent_sample = 40

    # Read in the data
    trainings_read_data = DecisionTree.Data.read_data(trainings_file)
    test_read_data = DecisionTree.Data.read_data(test_file)

    # Create an Data object of the Data
    trainings_data = DecisionTree.Data(trainings_read_data[0], trainings_read_data[1])
    test_data = DecisionTree.Data(test_read_data[0], test_read_data[1])

    first_forest = Forest(data=trainings_data, number_trees=number_trees, percentage_samples=percent_sample, number_attributes=number_attr, max_depth=max_depth)
    first_forest.train()
    predicted_labels = first_forest.test(test_data)
    print(test_data.print_confusionmatrix(predicted_labels, trainings_data.labels))


class Forest:

    def __init__(self, data, number_trees, percentage_samples, number_attributes, max_depth):
        self.number_trees = number_trees
        self.data = data
        self.trees = list()
        self.percentage_samples = percentage_samples
        self.number_attributes = number_attributes
        self.max_depth = max_depth

    def train(self):
        for x in range(self.number_trees):
            random_samples = self.data.random_sample(self.percentage_samples)
            random_data = DecisionTree.Data(random_samples[0], random_samples[1])
            tree = DecisionTree.Tree(random_data, self.max_depth)
            tree.train_forest(self.number_attributes)
            self.trees.append(tree)

    def test(self, data):
        predicted_labels = list()
        for sample in data.attributes:
            majority_vote_list = list()
            for tree in self.trees:
                majority_vote_list.append(tree.predcit_instance(sample, tree.root))
            predicted_labels.append(self.majority_vote(majority_vote_list))

        return predicted_labels

    def majority_vote(self, label_list):
        return Counter(label_list).most_common(1)[0][0]

if __name__ == "__main__":
    main()



