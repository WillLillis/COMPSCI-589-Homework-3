from copy import deepcopy
import csv
from random import randrange
from math import log2
import random
from statistics import stdev
#import matplotlib.pyplot as plt
from math import sqrt

# Need to continue work for numerical attributes...
# Next on the list is how to store that information in the tree's nodes correctly
    # Good way to do this?
    # Turn self.attr_val into a dict?
        # "type" indicates "categorical" or "numerical"
        # "value" holds:
            # if categorical, just hold the value
            # if numerical, hold 'leq' or 'geq'

class decision_tree:
    # data - The dataset passed in to create further nodes with
    # depth - recursive depth in terms of '\t' characters, used for debugging purposes
    # is_root - indicates whether a node is the root node, must be passed in as true for 
    #           the initial call, false for the other calls
    # classification - has argument passed in only when the caller is passing it an 
    #                  empty data set, indicates what classification (0 or 1 in the case) to make the resulting leaf node
    def __init__(self, data: list, attr_val, stopping_criteria: str, attr_type: list, attr_labels: list, \
        attr_vals: dict, depth = '', is_root = False, classification = None, split_metric="Info_Gain"):
        self.children = []
        self.threshold = None # holds threshold if we end up splitting based off of a numerical attribute
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

        self.is_leaf = False 
        self.classification = None # node isn't a leaf, so it doesn't decide what class a given instance belongs to 
        split_attr = None

        # evaluate stopping criteria
        if stopping_criteria == "minimal_size_for_split_criterion":
            thresh = 15 # empirical threshold
            if (len(data) - 1) <= thresh:
                self.is_leaf = True
                self.node_attr = None
                self.classification = get_majority_class(data)
                return
            else:
                self.is_leaf = False
        elif stopping_criteria == "minimal_gain_criterion":
             thresh = 0.1 # arbitary threshold
             info_gains = {} # information gain for each attribute
             data_set_only_labels = [] # strip off just the class labels (0's and 1's) to calculate entropy/ info gain
             for index in range(1, len(data)): # skip the first row (attribute labels)
                 data_set_only_labels.append(deepcopy(data[index][-1]))

             # get random subset of dataset's attributes
             # BUGBUG get_rand_cats is returning garbage
             attributes = get_rand_cats(deepcopy(attr_labels))
             print(f"ATTRIBUTES: {attributes}")
             for i in range(len(attributes)):
                 if attr_type[i] == True: # if it's a numerical attribute...
                     partitions, _ = partition_data_numerical(data, attributes[i], attr_labels) # paritition 'data' according to the current attribute 'attr'
                 else: # otherwise it's categorical
                     #BUGBUG need to pass in different value instead of attributes[i] or fix the function
                     partitions = partition_data_categorical(data, attributes[i], attr_vals, attr_labels) # paritition 'data' according to the current attribute 'attr'
                 #BUGBUG info_gains getting assigned float values here!!!!
                 info_gains[attributes[i]] = info_gain(data_set_only_labels, partitions) 
             split_attr = max(info_gains, key = info_gains.get) # get the attribute of maximal gain
             print(f"LOOK HERE!!!Split attr: {split_attr}")
             print(f"info_gains: {info_gains}")
             if info_gains[split_attr] < thresh:
                self.is_leaf = True
                self.node_attr = None
                self.classification = get_majority_class(data)
                return
             else:
                self.is_leaf = False
                self.node_attr = split_attr
        elif stopping_criteria == "maximal_depth_stopping_criterion":
            thresh = 10
            if len(depth) >= thresh:
                self.is_leaf = True
                self.node_attr = None
                self.classification = get_majority_class(data)
                return
            else:
                self.is_leaf = False
        else:
            print(f"Error! Invalid stopping criteria argument provided! ({stopping_criteria})")

        # find best attribute to split off of based off of Information Gain/ Gini Metric
        if split_metric == "Info_Gain":
            # if we're using minimal gain as our stopping criteria, everything should already be calculated!
            if stopping_criteria == "minimal_gain_criterion": 
                pass
            # otherwise calculate info gain as normal
            else:
                info_gains = {} # information gain for each attribute
                data_set_only_labels = [] # strip off just the class labels (0's and 1's) to calculate entropy/ info gain
                for index in range(1, len(data)): # skip the first row (attribute labels)
                    data_set_only_labels.append(deepcopy(data[index][-1]))

                # get random subset of dataset's attributes
                attributes = get_rand_cats(deepcopy(attr_labels))
                for attr in attributes:
                     if attr_type[attr] == True: # if it's a numerical attribute...
                         partitions, _ = partition_data_numerical(data, attr, attr_labels) # paritition 'data' according to the current attribute 'attr'
                     else: # otherwise it's categorical
                         partitions = partition_data_categorical(data, attr, attr_vals, attr_labels) # paritition 'data' according to the current attribute 'attr'
                     info_gains[attr] = info_gain(data_set_only_labels, partitions)
                split_attr = max(info_gains, key = info_gains.get) # get the attribute of maximal gain
                #print(f'{depth}Splitting based off of attribute {split_attr}, gain: {info_gains[split_attr]}')
        elif split_metric == "Gini":
            ginis = {}
            data_set_only_labels = []
            for index in range(1, len(data)): # skip the first row (attribute labels)
                data_set_only_labels.append(deepcopy(data[index][-1]))

            # get random subset of dataset's attributes
            attributes = get_rand_cats(deepcopy(attr_labels))
            for attr in attributes:
                partitions = partition_data_categorical(data, attr, attr_vals, attr_labels) # paritition 'data' according to the current attribute 'attr'
            split_attr = min(ginis, key = ginis.get)
        else:
            print("ERROR: Invalid split metric supplied!")
            return
        
        #BUGBUG split_attr got a float value somehow?
        self.node_attr = split_attr
        # partition data based off of split_attr
        if attr_type[i] == True: # if it's a numerical attribute...
            child_data, self.threshold = partition_data_numerical(data, split_attr, attr_labels, labels_only=False) # paritition 'data' according to the current attribute 'attr'
        else: # otherwise it's categorical
            child_data = partition_data_categorical(data, split_attr, attr_vals, attr_labels, labels_only=False) # paritition 'data' according to the current attribute 'attr'
        
        # translate split_attr to an index in attr_type for if block below...
        #BUGBUG Don't have access to data labels via data[0]
        for i in range(len(attr_type)):
            if attr_labels[i] == split_attr:
                split_attr_index = i
                break

        if attr_type[split_attr_index] == True: # if it's numerical...
            # BUGBUG need to redo this for numerical attributes!
            for i in range(2): # with numerical attributes there's always 2 partitions
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
        else: # otherwise it's categorical
            #for i in range(len(attr_vals[split_attr])):
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
        # BUGBUG fix this for numerical attributes, probably need a completely separate section
        if attr_type[split_attr_index] == True: # if it's numerical...
            # create dictionary self.attr_val here 
            # could be smarter about this, but I'm just going to hard code it...
            tmp_attr_val = {}
            tmp_attr_val["type"] = "numerical"
            tmp_attr_val["value"] = "leq" # less than or equal to partition
            if len(child_data[0]) > 1:
                self.children.append(decision_tree(child_data[0], tmp_attr_val, stopping_criteria, attr_type, attr_labels,\
                    attr_vals, depth=depth + '\t', split_metric=split_metric))
            else:
                self.children.append(decision_tree(child_data[0], tmp_attr_val, stopping_criteria, attr_type, attr_labels,\
                    attr_vals, depth=depth + '\t', classification=majority, split_metric=split_metric))
            tmp_attr_val["value"] = "g" # greater than partition
            if len(child_data[1]) > 1:
                self.children.append(decision_tree(child_data[1], tmp_attr_val, stopping_criteria, attr_type, attr_labels,\
                    attr_vals, depth=depth + '\t', split_metric=split_metric))
            else:
                self.children.append(decision_tree(child_data[1], tmp_attr_val, stopping_criteria, attr_type, attr_labels,\
                    attr_vals, depth=depth + '\t', classification=majority, split_metric=split_metric))
        else: # otherwise it's categorical
            # Need to support dictionaries here...
            tmp_attr_val = {}
            tmp_attr_val["type"] = "categorical"
            for i in range(len(child_data)):
                tmp_attr_val["value"] = i
                if len(child_data[i]) > 1:#if len(child_data[i]) != 0: 
                    self.children.append(decision_tree(child_data[i], i, stopping_criteria, attr_type, attr_labels,\
                        attr_vals, depth=depth + '\t', split_metric=split_metric))
                else:
                    self.children.append(decision_tree(child_data[i], i, stopping_criteria, attr_type, attr_labels,\
                        attr_vals, depth=depth + '\t', classification=majority, split_metric=split_metric))
    
    # for debugging, conducts a DFS of the tree, printing out its attributes
    def recursive_print(self, depth=''):
        print(f'{depth}self.attr_val: {self.attr_val}')
        print(f'{depth}self.is_leaf: {self.is_leaf}')
        print(f'{depth}self.node_attr: {self.node_attr}')
        print(f'{depth}self.classification: {self.classification}')
        for child in self.children:
            child.recursive_print(depth + '\t')
    
    # classifies data 'instance' using the current tree
    def classify_instance(self, instance, attr_to_index, attr_type):
        #print("CLASSIFY INSTANCE!")
        #print(f"instance: {instance}")
        #print(f"attr_to_index: {attr_to_index}")
        #print(f"attr_type: {attr_type}")
        if self.is_leaf == True: # base case
            print(f"Base case: returning {self.classification}")
            return self.classification
        #print(f"attr_to_index: {attr_to_index}")
        #print(f"self.node_attr: {self.node_attr}")
        #BUGBUG stringifying self.node_attr fixes the key error but may break other things
        #if attr_type[attr_to_index[str(self.node_attr)]] == True: # if it's numerical...
        if attr_type[self.node_attr] == True:
        #if False:
            #if instance[attr_to_index[str(self.node_attr)]] <= self.threshold:
            if instance[self.node_attr] <= self.threshold:
                return self.children[0].classify_instance(instance, attr_to_index, attr_type)
            else:
                return self.children[1].classify_instance(instance, attr_to_index, attr_type)
        else: # otherwise it's categorical
            # this is legitimately the worst code I've ever written in my life but at this point
            # I have another midterm to study for and don't care, it's working
            match self.node_attr:
                case 0:
                    tmp = '\ufeff#handicapped-infants'
                case 1:
                    tmp = 'water-project-cost-sharing'
                case 2:
                    tmp = 'adoption-of-the-budget-resolution'
                case 3:
                    tmp = 'physician-fee-freeze'
                case 4:
                    tmp = 'el-salvador-adi'
                case 5:
                    tmp = 'religious-groups-in-schools'
                case 6:
                    tmp = 'anti-satellite-test-ban'
                case 7:
                    tmp = 'aid-to-nicaraguan-contras'
                case 8:
                    tmp = 'mx-missile'
                case 9:
                    tmp = 'immigration'
                case 10:
                    tmp = 'synfuels-corporation-cutback'
                case 11:
                    tmp = 'education-spending'
                case 12:
                    tmp = 'superfund-right-to-sue'
                case 13:
                    tmp = 'crime'
                case 14:
                    tmp = 'duty-free-exports'
                case 15:
                    tmp = 'export-administration-act-south-africa'
            for child in self.children:
                #if child.attr_val == instance[cat_to_index(self.node_attr)]: # get instance's value for the current node's 'self.node_attr'-> has to match in value with one of children
                # need to translate self.node_attr which is a number
                #if self.node_attr in attr_to_index:
                # trying out replacing self.node_attr with tmp
                if tmp in attr_to_index:
                    new_var = child.attr_val
                    print(f"If {new_var} == {instance[attr_to_index[tmp]]}")
                    if child.attr_val == instance[attr_to_index[tmp]]: # get instance's value for the current node's 'self.node_attr'-> has to match in value with one of children
                        print("Recursing!")
                        return child.classify_instance(instance, attr_to_index, attr_type)
                else:
                    print(f'BAD ATTRIBUTE LABEL! ({self.node_attr}, {tmp})')
                    return None
            print("GOT HERE")

# partition dataset 'data' based off of attribute 'attr'
# if labels_only=True-> returns partitions ONLY WITH CLASS LABELS (0's and 1's, that's it)
# if labels_only=False-> returns the partitions with entire rows from data set copied in
    # in this case, each partition gets a row of attribute labels at the top
# pass in attr to index dict?
def partition_data_categorical(data, attr, attr_vals: dict, attr_labels: list, labels_only=True)->list: 
    print("PARTITION!!!!")
    print(f"attr: {attr}")
    print(f"attr_vals: {attr_vals}")
    print(f"attr_labels: {attr_labels}")

    partitions = [] # creating multi-dimensional arrays in python is weird...
    #BUGBUG stringifying attr may break stuff
    #for i in range(len(attr_vals[str(attr)])):
    for i in range(3):
        partitions.append([])

    #print(f"partitions: {partitions}")
    #BUGBUG bad attr_index
    #for i in range(len(data[0])):
    #    if data[0][i] == attr:
    #        attr_index = i
    #        break
    for i in range(len(attr_labels) - 1):
        if attr_labels[i] == attr:
            attr_index = i
            break
    # going to abuse the fact that the attribute values are 0,1,2 and use them as indices
        # using value of attribute for index into partition list (array?)
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

# BUGBUG have this return the average value used for the split????
def partition_data_numerical(data, attr, attr_labels: list, labels_only=True)->list:
    # always split to two classes based on threshold with numerical attributes...
    partitions = []
    partitions.append([]) # <= partition
    partitions.append([]) # > partition

    # for now we'll just use the "average" approach, will go back and try out the in between approach later
    #BUGBUG don't use data[0]
    for i in range(len(attr_labels) - 1):
        if attr_labels[i] == attr:
            attr_index = i
            break
    # grab the average value....
    avg = 0
    for i in range(1, len(data)): # skip the first row
        avg += data[i][attr_index]
    avg /= (len(data) - 1) # could check before we potentially divide by 0....

    if labels_only == True:
        for i in range(1, len(data)): # skip the first row
            if data[i][attr_index] <= avg:
                partitions[0].append(deepcopy(data[i][-1]))
            else:
                partitions[1].append(deepcopy(data[i][-1]))
    else:
        for i in range(1, len(data)): # skip the first row
            if data[i][attr_index] <= avg:
                partitions[0].append(deepcopy(data[i]))
            else:
                partitions[1].append(deepcopy(data[i]))

    return partitions, avg


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

# just needs to return list of attribute labels
# could also return indices?
# just pass in the first row of the dataset with the labels AS A DEEPCOPY
    # this function will destructively modify its `cats` parameter
# assuming the classification label is included here just for simplicity
# if num_cats_req is specified, we'll use that number
# otherwise we'll use the sqrt heuristic
def get_rand_cats(cats: list, num_cats_req=0):
    ret_cats = list()
    num_cats = 0
    if num_cats_req <= 0:
        num_cats = max(int(sqrt(len(cats) - 1)), 1)
    else:
        num_cats = num_cats_req
        while num_cats > (len(cats) - 1):
            num_cats -= 1
    for _ in range(num_cats):
        ret_cats.append(cats.pop(random.randrange(len(cats) - 1)))
    return ret_cats

# for simplicity we'll assume the labels occupy the first row, so we'll ignore that
def get_majority_class(data: list):
    if len(data) < 2:
        print("ERROR: Bad dataset!")
        return None
    counts = {}
    for i in range(1, len(data)):
        if data[i][-1] in counts:
            counts[data[i][-1]] += 1
        else:
            counts[data[i][-1]] = 1
    return max(counts, key=counts.get)

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
    pass
    #test_set_acc = []
    #training_set_acc = []
    #for _ in range(100):
    #    training_set, test_set, cat_to_attr_index = prepare_data()
    #    tree = decision_tree(training_set, None, '', is_root=True, split_metric="Gini")
    #    #tree = decision_tree(training_set, None, '', is_root=True, extra_stop=True)
    #    #tree.recursive_print()
    #    num_correct = 0
    #    for example in test_set:
    #        if tree.classify_instance(example, cat_to_attr_index) == example[-1]:
    #            num_correct += 1
    #    #print(f'Test set score: {num_correct / len(test_set)}')
    #    test_set_acc.append(num_correct / len(test_set))

    #    num_correct = 0
    #    for index in range(1, len(training_set)): # skip the first row (attribute labels)
    #        if tree.classify_instance(training_set[index], cat_to_attr_index) == training_set[index][-1]:
    #            num_correct += 1
    #    #print(f'Training set score: {num_correct / len(training_set)}')
    #    training_set_acc.append(num_correct / len(training_set))

    #print(f'Test set average score: {sum(test_set_acc) / len(test_set_acc)}, stdev: {stdev(test_set_acc)}')
    #print(f'Training set average score: {sum(training_set_acc) / len(training_set_acc)}, stdev: {stdev(training_set_acc)}')

    #training_plot = plt.figure(1)
    #plt.hist(training_set_acc, 10, ec="k")
    #plt.xlabel("Accuracy (%)")
    #plt.ylabel("Counts/Bin")
    #plt.title("Training Set Accuracy")
    #training_plot.show()

    #test_plot = plt.figure(2)
    #plt.hist(test_set_acc, 10, ec="k")
    #plt.xlabel("Accuracy (%)")
    #plt.ylabel("Counts/Bin")
    #plt.title("Test Set Accuracy")
    #test_plot.show()

    

    #input()

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