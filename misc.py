# in an effort to keep this project more manageable, I'll reorganize and throw
# some of the utility functions in here...

# assignment asks us to define a boostrap method to pass data to the trees of the random forest
# assignment then says we'll test the forest via kfold cross validation, so I guess we don't have to keep track of 
# our "out of bag" instances???
from copy import deepcopy
import csv
import random

# returns a bootstrap of the data set passed in, with labels still in the first row
def bootstrap(data: list):
    length = len(data)
    strap = list()
    strap.append(deepcopy(data[0])) # keep the labels up top

    for _ in range(length - 1):
        strap.append(deepcopy(data[random.randrange(1, length - 1)]))
    return strap

# generates the 
def k_folds_gen(k: int, file_name: str):
    # find class proportiions in data set
    # make k folds
    # populate each fold according to class proportions (randomly)
    # for each fold i...
        # pass training data in to train random forest
        # evaluate on fold i

    with open(file_name, encoding="utf-8") as raw_data_file:
        # the data files all follow different labeling conventions and/or use different delimiters...
        # could make this more general, but here we'll more or less hardcode in the correct procedure for
        # the cancer, house votes, and wine datasets
        if 'hw3_cancer.csv' in file_name:
            data_reader = csv.reader(raw_data_file, delimiter='\t')
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                for j in range(len(data_set[i])):
                    data_set[i][j] = int(float(data_set[i][j]))
        elif 'hw3_house_votes_84.csv' in file_name:
            data_reader = csv.reader(raw_data_file)
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                for j in range(len(data_set[i])):
                    data_set[i][j] = int(data_set[i][j])
        elif 'hw3_wine.csv' in file_name:
            data_reader = csv.reader(raw_data_file, delimiter='\t')
            data_set = list(data_reader)
            for i in range(1, len(data_set)):
                data_set[i][0] = int(data_set[i][0])
                for j in range(1, len(data_set[i])):
                    data_set[i][j] = float(data_set[i][j])
            # for the sake of simplicity, at this point I'm going to move
            # the wine classes to the last column so it matches the other data sets
            for entry in data_set:
                tmp = entry.pop(0)
                entry.append(tmp)

    class_partitioned = {}
    for i in range(1, len(data_set)):
        if data_set[i][-1] in class_partitioned:
            class_partitioned[data_set[i][-1]].append(data_set[i])
        else:
            class_partitioned[data_set[i][-1]] = list()
            class_partitioned[data_set[i][-1]].append(data_set[i])

    #print(class_partitioned)
    class_proportions = {}
    for item in class_partitioned:
        class_proportions[item] = len(class_partitioned[item]) / (len(data_set) - 1)
    

    #for entry in class_partitioned:
    #    print(len(class_partitioned[entry]))
    #for entry in class_proportions:
    #    print(class_proportions[entry])
    #for entry in class_partitioned:
    #    print(len(class_partitioned[entry]))
    #print(class_proportions)
    

    # create list of lists to hold our k folds
    k_folds = []
    for _ in range(k):
        k_folds.append([])

    entries_per_fold = int((len(data_set) - 1) / k)
    while k * entries_per_fold > (len(data_set) - 1):
        entries_per_fold -= 1

    if len(class_proportions) == 2:
        for index in range(k):
            for _ in range(entries_per_fold):
                if random.uniform(0,1) <= class_proportions[0]:
                    if len(class_partitioned[0]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[0]))
                    new_entry = class_partitioned[0].pop(tmp)
                    k_folds[index].append(new_entry)
                else:
                    if len(class_partitioned[1]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[1]))
                    new_entry = class_partitioned[1].pop(tmp)
                    k_folds[index].append(new_entry)
    elif len(class_proportions) == 3:
        for index in range(k):
            for _ in range(entries_per_fold):
                u = random.uniform(0,1)
                if u <= class_proportions[1]:
                    if len(class_partitioned[1]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[1]))
                    new_entry = class_partitioned[1].pop(tmp)
                    k_folds[index].append(new_entry)
                elif (u > class_proportions[1]) and (u <= (class_proportions[1] + class_proportions[2])):
                    if len(class_partitioned[2]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[2]))
                    new_entry = class_partitioned[2].pop(tmp)
                    k_folds[index].append(new_entry)
                else:
                    if len(class_partitioned[3]) == 0:
                        break
                    tmp = random.randrange(len(class_partitioned[3]))
                    new_entry = class_partitioned[3].pop(tmp)
                    k_folds[index].append(new_entry)
    else:
        print("ERROR!!!!!!!")

    return k_folds

    # populate the folds according to the original data set's class proportions
    # ok to do this in a randomized fashion?

    
    