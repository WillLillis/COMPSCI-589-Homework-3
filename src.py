from copy import deepcopy
import random_forest
from decision_tree import prepare_data
from misc import bootstrap
from misc import k_folds_gen

# TODO 
    # Clean up existing decision tree code
    # Add option for numeric attributes
    # Choose + implement new stopping criteria
    # Write bootstrapping routine to create synthetic datasets
    # Actual Random Forest implementation
        # majority voting mechanism
    # Implement stratified cross-validation technique 
    # handle categorical vs. numerical attributes in attr_vals dict
        # instead of attr->list, instead hold
        # attr->list
        #     ->bool (indicating numeric or not)
        # if numberic is True, list should empty!


def main():
    k_folds, attr_type, data_labels = k_folds_gen(10, 'hw3_house_votes_84.csv')
    print('hw3_house_votes_84.csv')
    for fold in k_folds:
        #print(f"New fold: {fold}\n\n")
        pass

    data = k_folds[1]
    for i in range(2, len(k_folds)):
        data += k_folds[i]
    
    # slap the labels back onto the top of the k_folds list of lists
    data.insert(0, data_labels)
    ##print(data)
    #k_folds, attr_type, data_labels = k_folds_gen(10, 'hw3_wine.csv')
    
    #print('hw3_wine.csv')
    #for fold in k_folds:
    #    #print(f"New fold: {fold}\n\n")
    #    pass

    data = k_folds[1]
    for i in range(2, len(k_folds)):
        data += k_folds[i]
    
    # slap the labels back onto the top of the k_folds list of lists
    # ONLY FOR WINE?
    #data.insert(0, data_labels)
    data_labels_num = deepcopy(data_labels)
    # vvvvv for wine vvvvv
    for i in range(len(data_labels_num) - 1):
        #data_labels_num[i] = int(data_labels_num[i])
        data_labels_num[i] = i
    # vvvvv for congress vvvvv
    #for i in range(len(data_labels_num) - 1):
    #    data_labels_num[i] = i

    #print(data)
    #BUGBUG check how data_labels behaves with the different data sets
    forest = random_forest.random_forest(data, 10, attr_type, data_labels_num)
    forest.recur_print()

    num_correct = 0
    num = 0
    for entry in k_folds[0]:
        #print(f"class: {entry[-1]}")
        num += 1
        #BUGBUG need to pass attr_to_index here
        if forest.classify_instance(entry, attr_type) == entry[-1]:
            print("Great success!")
            num_correct += 1
        else:
            print("Failure!")
    print(f"Score: {num_correct / num}")

    print(f"attr_type: {attr_type}")

if __name__ == "__main__":
    main()