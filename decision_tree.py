from copy import deepcopy
import csv
from random import randrange
from math import log2
from statistics import stdev
import matplotlib.pyplot as plt

# stopping criteria, can pick one or combine
    # minimal_size_for_split_criterion : data set < n instances, pick n empirically
    # minimal_gain_criterion : stop splitting once info gain is too low, pick threshold empirically
    # maximal_depth_stopping_criterion : stop when some max depth is reached, pick depth empirically

# stopping criteria arg->just use strings for clarity like split_metric?

class decision_tree:
    # data - The dataset passed in to create further nodes with
    # depth - recursive depth in terms of '\t' characters, used for debugging purposes
    # is_root - indicates whether a node is the root node, must be passed in as true for 
    #           the initial call, false for the other calls
    # classification - has argument passed in only when the caller is passing it an 
    #                  empty data set, indicates what classification (0 or 1 in the case) to make the resulting leaf node
    def __init__(self, data, attr_val, depth = '', is_root = False, classification = None, split_metric="Info_Gain", extra_stop=False):
        self.children = []
        if is_root == True:
            self.attr_val = None 
        else:
            self.attr_val = attr_val
        # first check if an empty dataset was passed in 
        if classification != None: # determined by caller, != None only when an empty dataset is passed in
            #print(f'{depth}Empty partition! Creating leaf node!')
            self.is_leaf = True
            self.node_attr = None
            self.classification = classification 
            return
        # next check if 'data' is homogeneous (all members of dataset belong to same class)
        homogeneous = True
        prev = data[1][-1] # first data entry label
        for index in range(2, len(data)): # data[0] has attribute labels, data[1] is used to initialize 'prev'-> start at index 2
            if prev != data[index][-1]:
                homogeneous = False
                break
        if homogeneous == True: # if the data set is homogenous, we're done
            #print(f'{depth}Homogenous! Creating leaf node!')
            self.is_leaf = True
            self.node_attr = None
            self.classification = data[1][-1]
            return

        # extra credit option here, check if 85%+ instances belong to the same class
        if extra_stop == True:
            counts = {}
            for index in range(1, len(data)): # skip attribute labels in first row
                if data[index][-1] in counts:
                    counts[data[index][-1]] += 1
                else:
                    counts[data[index][-1]] = 1
            for item in counts:
                if counts[item] >= (0.85 * (len(data) - 1)):
                    self.is_leaf = True
                    self.node_attr = None
                    self.classification = item
                    return
        
        # if it's not homogenous...
        #print(f'{depth}Not homogenous! Continuing...')
        self.is_leaf = False 
        self.classification = None # node isn't a leaf, so it doesn't decide what class a given instance belongs to 

        split_attr = None
        # find best attribute to split off of based off of Information Gain/ Gini Metric
        if split_metric == "Info_Gain":
            info_gains = {} # information gain for each attribute
            data_set_only_labels = [] # strip off just the class labels (0's and 1's) to calculate entropy/ info gain
            for index in range(1, len(data)): # skip the first row (attribute labels)
                data_set_only_labels.append(deepcopy(data[index][-1]))

            for attr_index in range(len(data[0]) - 1): # -1 there to skip the 'target' attribute
                partitions = partition_data(data, data[0][attr_index]) # paritition 'data' according to the current attribute 'attr'
                info_gains[data[0][attr_index]] = info_gain(data_set_only_labels, partitions) 
                #print(f'{depth}information gain of split based off of {data[0][attr_index]} is {info_gains[data[0][attr_index]]}')
            split_attr = max(info_gains, key = info_gains.get) # get the attribute of maximal gain
            #print(f'{depth}Splitting based off of attribute {split_attr}, gain: {info_gains[split_attr]}')
        elif split_metric == "Gini":
            ginis = {}
            data_set_only_labels = []
            for index in range(1, len(data)): # skip the first row (attribute labels)
                data_set_only_labels.append(deepcopy(data[index][-1]))

            for attr_index in range(len(data[0]) - 1): # -1 there to skip the 'target' attribute
                partitions = partition_data(data, data[0][attr_index]) # paritition 'data' according to the current attribute 'attr'
                ginis[data[0][attr_index]] = 0
                for partition in partitions:
                    ginis[data[0][attr_index]] += (len(partition) / len(data)) * gini_criterion(partition)
                #print(f'{depth}gini criterion of split based off of {data[0][attr_index]} is {ginis[data[0][attr_index]]}')
            split_attr = min(ginis, key = ginis.get)
        else:
            print("ERROR: Invalid split metric supplied!")
            return
        

        self.node_attr = split_attr
        # partition data based off of split_attr
        child_data = partition_data(data, split_attr, labels_only=False)

        # calculate dataset majority in case of empty partition
        # hardcoded '3' here is a little dumb, could make this more dynamic
        for i in range(3):
            if len(child_data[i]) <= 1:
                num_zero = 0
                num_one = 0
                for instance in data:
                    if instance[-1] == 0:
                        num_zero +=  1
                    else:
                        num_one += 1
                majority = 0 if num_zero >= num_one else 1
                break

        for i in range(3):
            if len(child_data[i]) > 1:#if len(child_data[i]) != 0: 
                self.children.append(decision_tree(child_data[i], i, depth=depth + '\t', split_metric=split_metric, extra_stop=extra_stop))
            else:
                self.children.append(decision_tree(child_data[i], i, depth=depth + '\t', classification=majority, split_metric=split_metric, extra_stop=extra_stop))
    
    # for debugging, conducts a DFS of the tree, printing out its attributes
    def recursive_print(self, depth=''):
        print(f'{depth}self.attr_val: {self.attr_val}')
        print(f'{depth}self.is_leaf: {self.is_leaf}')
        print(f'{depth}self.node_attr: {self.node_attr}')
        print(f'{depth}self.classification: {self.classification}')
        for child in self.children:
            child.recursive_print(depth + '\t')
    
    # classifies data 'instance' using the current tree
    def classify_instance(self, instance, attr_to_index):
        if self.is_leaf == True: # base case
            return self.classification
        for child in self.children:
            #if child.attr_val == instance[cat_to_index(self.node_attr)]: # get instance's value for the current node's 'self.node_attr'-> has to match in value with one of children
            if self.node_attr in attr_to_index:
                if child.attr_val == instance[attr_to_index[self.node_attr]]: # get instance's value for the current node's 'self.node_attr'-> has to match in value with one of children
                    return child.classify_instance(instance, attr_to_index)
            else:
                print(f'BAD ATTRIBUTE LABEL! ({self.node_attr})')
                return "Error: Bad attribute label used for split"

# partition dataset 'data' based off of attribute 'attr'
# if labels_only=True-> returns partitions ONLY WITH CLASS LABELS (0's and 1's, that's it)
# if labels_only=False-> returns the partitions with entire rows from data set copied in
    # in this case, each partition gets a row of attribute labels at the top
def partition_data(data, attr, labels_only=True):
    partitions = [] # creating multi-dimensional arrays in python is weird...
    partitions.append([])
    partitions.append([])
    partitions.append([])
    for i in range(len(data[0])):
        if data[0][i] == attr:
            attr_index = i
            break
    # going to abuse the fact that the attribute values are 0,1,2 and use them as indices
    # if they weren't I could just use a dict to map the value to an index, but that looks messy....
    if labels_only == True:
         for i in range(1, len(data)): # skip the first row
            partitions[data[i][attr_index]].append(deepcopy(data[i][-1]))
    else: #BUGBUG need to take out attribute we're partitioning based off of? would give info gain of 0 so maybe not
        for partition in partitions: # each partition needs labels up top...
            partition.append(deepcopy(data[0]))
        for i in range(1, len(data)):
            partitions[data[i][attr_index]].append(deepcopy(data[i]))

    return partitions

# only pass in class labels
def entropy(data_set):
    counts = {}
    entropy = 0
    set_size = len(data_set)
    if set_size == 0: # special case, the empty set has an entropy of 0 by definition
        return 0
    
    # count how many of each value we have in 'data_set'
    for entry in data_set:
        if entry in counts: # if it's already been added to the dict, increment the count
            counts[entry] += 1
        else: # otherwise add it to the dict with an initial value of 1
            counts[entry] = 1

    for label in counts:
        entropy += (-counts[label] / set_size) * log2(counts[label] / set_size) # entropy formula...
    return entropy

# orig_data is the original data set, data_split is a tuple of that set partitioned according to some attribute
# assuming we're just using average entropy here...
def info_gain(orig_data, data_split):
    entropies = []
    for split in data_split:
        entropies.append((entropy(split), len(split)))
    
    avg_entropy = 0
    for entry in entropies:
        avg_entropy += (entry[1] / len(orig_data)) * entry[0]  # weighted average-> (size of partition/ size of set) * entropy of partition

    return entropy(orig_data) - avg_entropy 

# only pass in class labels
# want minimum gini criterion      
def gini_criterion(labels):
    counts = {}
    for x in labels:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1

    accum = 0
    for x in counts:
        accum += (x/len(labels))**2

    return 1 - accum

# read in the data from the csv and randomly split into training (80%) and test (20%) sets, return these sets
def prepare_data(file_name: str):
    # load in data
    data_set = [] # entire dataset, as read in from the csv
    training_set = [] # training set partition of the entire data set
    test_set = [] # testing partition of the entire data set
    cat_to_attr_label = {}
    num_attr = 0
    num_entries = 0

    with open(file_name, encoding="utf-8") as raw_data_file:
        data_reader = csv.reader(raw_data_file)
        data_set = list(data_reader)
    
    # grab some general info about the data set...
    # see how many attributes are in the data set
    for item in data_set[0]:
        num_attr += 1
    num_attr -= 1 # just assume the last item in a row is the label....
    num_entries = len(data_set) - 1 # don't count the attribute labels up top
    
    # sanity checks
    if num_attr <= 0 or num_entries == 0:
        print("Error: Bad data")
        return
    else:
        #print(f"Dataset has {num_entries} entries with {num_attr} attributes.")
        pass

    # I imagine this doesn't generalize to other data sets, but for this one we'll cast all of the attributes to ints
    for entry in range(1, num_entries + 1):
        for attr in range(num_attr + 1): # also want to cast the class labels as ints
            data_set[entry][attr] = int(data_set[entry][attr])

    for index in range(len(data_set[0])):
        cat_to_attr_label[data_set[0][index]] = index
    #for labels in data_set[0]:
    #    cat_to_attr_label[]
    # split between training (80%) and testing data (20%)
    training_set_size = int(num_entries * 0.8)
    test_set_size = int(num_entries * 0.2)
    # deal with any potential rounding issues
    while (training_set_size + test_set_size) > num_entries:
        test_set_size -= 1
    while (training_set_size + test_set_size) < num_entries:
        training_set_size += 1
    
    training_set.append(data_set.pop(0)) # need class labels at the top of the training set
    # randomly select the entries to go into the training set, note this selection method also randomizes the order
    for _ in range(training_set_size):
        training_set.append(data_set.pop(randrange(num_entries)))
        num_entries -= 1

    test_set = deepcopy(data_set) # remaining entries become the test set

    return training_set, test_set, cat_to_attr_label

def main():
    test_set_acc = []
    training_set_acc = []
    for _ in range(100):
        training_set, test_set, cat_to_attr_index = prepare_data()
        tree = decision_tree(training_set, None, '', is_root=True, split_metric="Gini")
        #tree = decision_tree(training_set, None, '', is_root=True, extra_stop=True)
        #tree.recursive_print()
        num_correct = 0
        for example in test_set:
            if tree.classify_instance(example, cat_to_attr_index) == example[-1]:
                num_correct += 1
        #print(f'Test set score: {num_correct / len(test_set)}')
        test_set_acc.append(num_correct / len(test_set))

        num_correct = 0
        for index in range(1, len(training_set)): # skip the first row (attribute labels)
            if tree.classify_instance(training_set[index], cat_to_attr_index) == training_set[index][-1]:
                num_correct += 1
        #print(f'Training set score: {num_correct / len(training_set)}')
        training_set_acc.append(num_correct / len(training_set))

    print(f'Test set average score: {sum(test_set_acc) / len(test_set_acc)}, stdev: {stdev(test_set_acc)}')
    print(f'Training set average score: {sum(training_set_acc) / len(training_set_acc)}, stdev: {stdev(training_set_acc)}')

    training_plot = plt.figure(1)
    plt.hist(training_set_acc, 10, ec="k")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Counts/Bin")
    plt.title("Training Set Accuracy")
    training_plot.show()

    test_plot = plt.figure(2)
    plt.hist(test_set_acc, 10, ec="k")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Counts/Bin")
    plt.title("Test Set Accuracy")
    test_plot.show()

    

    input()

if __name__ == '__main__':
    main()
    #with open('house_votes_84.csv', encoding="utf-8") as raw_data_file:
    #    data_reader = csv.reader(raw_data_file)
    #    data_set = list(data_reader)
    
    ## grab some general info about the data set...
    ## see how many attributes are in the data set
    #num_attr = 0
    #num_entries = 0
    #cat_to_attr_label = {}
    #for item in data_set[0]:
    #    num_attr += 1
    #num_attr -= 1 # just assume the last item in a row is the label....
    #num_entries = len(data_set) - 1 # don't count the attribute labels up top

    ## I imagine this doesn't generalize to other data sets, but for this one we'll cast all of the attributes to ints
    #for entry in range(1, num_entries + 1):
    #    for attr in range(num_attr + 1): # also want to cast the class labels as ints
    #        data_set[entry][attr] = int(data_set[entry][attr])

    #for index in range(len(data_set[0])):
    #    cat_to_attr_label[data_set[0][index]] = index
    
    #tree = decision_tree(data_set, None, '', is_root=True)
    #tree.recursive_print()