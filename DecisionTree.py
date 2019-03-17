import sys
from collections import Counter
import random

def main():
    test_file = "Data/" + sys.argv[2]
    trainings_file = "Data/" + sys.argv[1]

    max_depth = 100

    # Set the hyper-parameters
    if "balance.scale" in trainings_file:
        max_depth = 3
    elif "nursery" in trainings_file:
        max_depth = 100
    elif "led" in trainings_file:
        max_depth = 100
    elif "synthetic.social" in trainings_file:
        max_depth = 100

    # Read in the data
    trainings_read_data = Data.read_data(trainings_file)
    test_read_data = Data.read_data(test_file)

    # Create Data objects out the data
    trainings_data = Data(trainings_read_data[0], trainings_read_data[1])
    test_data = Data(test_read_data[0], test_read_data[1])

    tree = Tree(data=trainings_data, max_depth=max_depth)
    tree.train()
    predicted_labels = tree.test(test_data)
    print(test_data.print_confusionmatrix(predicted_labels, trainings_data.labels))

# This class represents the data (attributes with values and the labels)
class Data:

    def __init__(self, attributes, labels):
        self.attributes = attributes
        self. labels = labels
        self.attributes_list = list(self.attributes[0].keys())

    # Returns a list of attr_values and labels
    @staticmethod
    def read_data(file_path):

        labels = list()
        att_val = dict()
        att_val_list = list()

        with open(file_path, 'r') as training_f:
            for line in training_f.readlines():
                line = line.strip()
                tmp_list = line.split(' ')
                # Add label to the label list
                labels.append(int(tmp_list[0]))
                for attr in range(1, len(tmp_list)):
                    # Split again the
                    tmp_str = tmp_list[attr].split(':')
                    att_val[int(tmp_str[0])] = int(tmp_str[1])
                att_val_list.append(att_val.copy())
                att_val.clear()

        return [att_val_list, labels]

    @staticmethod
    def compute_gini_index(labels):
        # Create a counter object to compute the relative frequencies
        class_counter = Counter(labels)
        length_dataset = len(labels)
        sum_classes = 0
        # Iterate over all classes
        for class_c in class_counter:
            sum_classes += ((class_counter[class_c] / length_dataset) ** 2)

        gini = (1 - sum_classes)

        return gini

    '''
    This method computes the gini index for attributes
    data is the data
    attribute is the attribute we want to compute
    '''
    @staticmethod
    def compute_gini_attribute(data, labels, attribute):
        len_data = len(data)
        # This is a list of all values of the attributes
        attribute_values_counter = Counter([d[attribute] for d in data])

        gini = 0
        # Compute the gini index for the attribute
        for value in attribute_values_counter:
            # list of indices of all the relevant elements
            attr_split_indices = [index for index, item in enumerate(data) if item[attribute] == value]
            # create a list of all relevant elements
            attr_split = [labels[index] for index in attr_split_indices]
            len_split = len(attr_split)
            gini += ((len_split / len_data) * Data.compute_gini_index(attr_split))

        return gini

    # Returns the attribute with the smallest gini index, attr_values: attributes with values
    @staticmethod
    def find_split(attr_values, labels, attributes):
        # List of all gini indices
        gini_indices = list()

        for attr in attributes:
            gini_indices.append(Data.compute_gini_attribute(attr_values, labels, attr))

        return attributes[gini_indices.index(min(gini_indices))]

    # Splits the data according to the best split
    @staticmethod
    def split_data(attr_values, labels, best_split):
        # Contains all the splits of attr_values
        attr_values_splits = list()

        # Conatins all the labels of the split
        labels_split = list()

        # List of all values
        values_list = set([d[best_split] for d in attr_values])

        for value in values_list:
            attr_values_splits_tmp = list()
            labels_split_tmp = list()
            for index, instance in enumerate(attr_values):
                if instance[best_split] == value:
                    attr_values_splits_tmp.append(instance)
                    labels_split_tmp.append(labels[index])
            attr_values_splits.append(attr_values_splits_tmp)
            labels_split.append(labels_split_tmp)

        return [attr_values_splits, labels_split]

    def random_sample(self, percent):
        # Compute the number which corresponds to the percantage given
        num = int(len(self.attributes) * (percent/100))

        # Create random indices of the
        # With replacement
        list_indices = random.sample(range(len(self.attributes)), num)
        # Without replacement
        #list_indices = random.choices(range(len(self.attributes)), k=num)

        attr_values_sample = [self.attributes[i] for i in list_indices]
        labels_sample = [self.labels[i] for i in list_indices]

        return [attr_values_sample, labels_sample]

    @staticmethod
    def random_attributes(attribute_list, number):
        if number >= len(attribute_list):
            number = len(attribute_list)

        return random.sample(attribute_list, number)

    @staticmethod
    def checkEqual(lst):
        return all(x == lst[0] for x in lst)

    def print_confusionmatrix(self, predicted_labels, training_data_labels):
        length = max(set(self.labels).union(set(training_data_labels)))
        confusionmatrix = [[0 for i in range(length)] for j in range(length)]

        for x in range(len(self.labels)):
            confusionmatrix[self.labels[x]-1][predicted_labels[x]-1] += 1

        output_str = ""
        for row in confusionmatrix:
            for entry in row:
                output_str += (str(entry) + " ")
            output_str = output_str[:-1]
            output_str += "\n"

        output_str = output_str[:-1]

        return output_str


class Node:

    # attr_split: attribute which splits at this node, value: value of the attribute,
    # children: list of all the children nodes
    def __init__(self, value):
        self.attr_split = None
        self.value = value
        self.children = list()
        self.label = None

    def train(self, attr_values, labels, attr_list, depth, max_depth):

        if depth >= max_depth:
            self.label = Counter(labels).most_common(1)[0][0]
            return

        # All samples for a given node belong to the same class
        if Data.checkEqual(labels):
            self.label = labels[0]
            return

        # There are no remaining attributes for further partitioning
        if len(attr_list) == 0:
            self.label = Counter(labels).most_common(1)[0][0]
            return

        # There are no samples left
        if len(attr_values) == 1:
            self.label = labels[0]
            return

        split_attr = Data.find_split(attr_values, labels, attr_list)
        # Set the split attribute of the node
        self.attr_split = split_attr
        attr_list2 = attr_list.copy()
        attr_list2.remove(split_attr)
        split = Data.split_data(attr_values, labels, split_attr)
        attr_values_split = split[0]
        labels_split = split[1]

        for attr_value, label in zip(attr_values_split, labels_split):
            node = Node(attr_value[0][split_attr])
            self.children.append(node)
            node.train(attr_value, label, attr_list2, depth+1, max_depth)

    def train_forest(self, attr_values, labels, attr_list, number_attr, depth, max_depth):

        if depth >= max_depth:
            self.label = Counter(labels).most_common(1)[0][0]
            return

        # All samples for a given node belong to the same class
        if Data.checkEqual(labels):
            self.label = labels[0]
            return

        # There are no remaining attributes for further partitioning
        if len(attr_list) == 0:
            self.label = Counter(labels).most_common(1)[0][0]
            return

        # There are no samples left
        if len(attr_values) == 1:
            self.label = labels[0]
            return

        random_attr_list = Data.random_attributes(attr_list, number_attr)
        split_attr = Data.find_split(attr_values, labels, random_attr_list)
        # Set the split attribute of the node
        self.attr_split = split_attr
        attr_list2 = attr_list.copy()
        attr_list2.remove(split_attr)
        split = Data.split_data(attr_values, labels, split_attr)
        attr_values_split = split[0]
        labels_split = split[1]

        for attr_value, label in zip(attr_values_split, labels_split):
            node = Node(attr_value[0][split_attr])
            self.children.append(node)
            node.train_forest(attr_value, label, attr_list2, number_attr, depth+1, max_depth)


class Tree:

    def __init__(self, data, max_depth):
        self.data = data
        # Previous
        self.attributes = data.attributes_list
        self.root = Node(None)
        self.max_depth = max_depth

    def train(self):
        self.root.train(self.data.attributes, self.data.labels, self.attributes, 0, self.max_depth)

    def train_forest(self, number_attr):
        self.root.train_forest(self.data.attributes, self.data.labels, self.attributes, number_attr, 0, self.max_depth)

    # This function takes the test data and returns a list with the classified samples
    def test(self, data):
        predicted_labels = list()
        for sample in data.attributes:
            # First split
            predicted_labels.append(self.predcit_instance(sample, self.root))

        return predicted_labels

    @staticmethod
    def compute_accuracy(data, prediction):

        count = 0
        for x in range(len(data.labels)):
            if data.labels[x] == prediction[x]:
                count += 1

        return (count/len(prediction))

    #
    def predcit_instance(self, instance, node):

        if node.label is not None:
            return node.label

        split = node.attr_split

        value_instance = instance[split]

        next_node = node.children[0]
        for child in node.children:
            if child.value == value_instance:
                next_node = child
                break

        return self.predcit_instance(instance, next_node)


if __name__ == "__main__":
    main()
