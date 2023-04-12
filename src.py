from copy import deepcopy
import random_forest
from decision_tree import prepare_data
from misc import bootstrap
from misc import k_folds_gen

# general testing purposes
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
    #print(data)
    #k_folds, attr_type, data_labels = k_folds_gen(10, 'hw3_wine.csv')
    
    #print('hw3_wine.csv')
    #for fold in k_folds:
    #    print(f"New fold: Size: {len(fold)}\n\n")
    #    pass

    #data = k_folds[1]
    #for i in range(2, len(k_folds)):
    #    data += k_folds[i]
    
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
            #print("Great success!")
            num_correct += 1
        else:
            #print("Failure!")
            pass
    print(f"Score: {num_correct / num}")

    print(f"attr_type: {attr_type}")

# wine dataset
def test_wine(num_trees: int, num_folds: int):
    k_folds, attr_type, data_labels = k_folds_gen(num_folds, 'hw3_wine.csv')
    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for k in range(num_folds):
        data = []
        test_fold = k_folds[k]
        for index in range(num_folds):
            if index != k:
                data += k_folds[index]
        # slap the labels back onto the top of the k_folds list of lists
        data.insert(0, data_labels)
        data_labels_num = deepcopy(data_labels)
        for i in range(len(data_labels_num) - 1):
            data_labels_num[i] = i

        forest = random_forest.random_forest(data, num_trees, attr_type, data_labels_num)

        accuracy1 = 0
        accuracy2 = 0
        accuracy3 = 0
        precision1 = 0
        precision2 = 0
        precision3 = 0
        recall1 = 0
        recall2 = 0
        recall3 = 0
        F1_1 = 0
        F1_2 = 0
        F1_3 = 0

        TP_1 = 0
        TP_2 = 0
        TP_3 = 0
        TN_1 = 0
        TN_2 = 0
        TN_3 = 0
        FP_1 = 0
        FP_2 = 0
        FP_3 = 0
        FN_1 = 0
        FN_2 = 0
        FN_3 = 0
        for entry in test_fold:
            output = forest.classify_instance(entry, attr_type)
            if entry[-1] == 1: # first class
                if output == 1:
                    TP_1 += 1
                    TN_2 += 1
                    TN_3 += 1
                elif output == 2:
                    FN_1 += 1
                    FP_2 += 1
                    TN_3 += 1
                elif output == 3:
                    FN_1 += 1
                    TN_2 += 1
                    FP_3 += 1
                else:
                    print("wtf")
            elif entry[-1] == 2: # second class
                if output == 1:
                    FP_1 += 1
                    FN_2 += 1
                    TN_3 += 1
                elif output == 2:
                    TN_1 += 1
                    TP_2 += 1
                    TN_3 += 1
                elif output == 3:
                    TN_1 += 1
                    FN_2 += 1
                    FP_3 += 1
                else:
                    print("wtf")
            elif entry[-1] == 3: # third class
                if output == 1:
                    FP_1 += 1
                    TN_2 += 1
                    FN_3 += 1
                elif output == 2:
                    TN_1 += 1
                    FP_2 += 1
                    FN_3 += 1
                elif output == 3:
                    TN_1 += 1
                    TN_2 += 1
                    TP_3 += 1
                else:
                    print("wtf")
        accuracy1 = (TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1)
        accuracy2 = (TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2)
        accuracy3 = (TP_3 + TN_3) / (TP_3 + TN_3 + FP_3 + FN_3)
        precision1 = (TP_1) / (TP_1 + FP_1)
        precision2 = (TP_2) / (TP_2 + FP_2)
        precision3 = (TP_3) / (TP_3 + FP_3)
        recall1 = (TP_1) / (TP_1 + FN_1)
        recall2 = (TP_2) / (TP_2 + FN_2)
        recall3 = (TP_3) / (TP_3 + FN_3)
        F1_1 = (2.0 * precision1 * recall1) / (precision1 + recall1)
        F1_2 = (2.0 * precision2 * recall2) / (precision2 + recall2)
        F1_3 = (2.0 * precision3 * recall3) / (precision3 + recall3)
    accuracies.append((accuracy1 + accuracy2 + accuracy3) / 3.0)
    precisions.append((precision1 + precision2 + precision3) / 3.0)
    recalls.append((recall1 + recall2 + recall3) / 3.0)
    F1s.append((F1_1 + F1_2 + F1_3) / 3.0)
    print(f"Wine Results ({num_trees} trees, {num_folds} folds):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

# house votes data set
def test_congress(num_trees: int, num_folds: int):
    k_folds, attr_type, data_labels = k_folds_gen(num_folds, 'hw3_house_votes_84.csv')
    accuracies = []
    precisions = []
    recalls = []
    F1s = []
    for k in range(num_folds):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        data = []
        test_fold = k_folds[k]
        for index in range(num_folds):
            if index != k:
                data += k_folds[index]
        data.insert(0, data_labels)
        data_labels_num = deepcopy(data_labels)
        for i in range(len(data_labels_num) - 1):
            data_labels_num[i] = i

        forest = random_forest.random_forest(data, num_trees, attr_type, data_labels_num)
        
        for entry in test_fold:
            output = forest.classify_instance(entry, attr_type)
            if entry[-1] == 0: # negative class instances
                if output == 0:
                    TN += 1
                elif output == 1:
                    FP += 1
                else:
                    print("wtf")
            elif entry[-1] == 1: # positive class instances
                if output == 0:
                    FN += 1
                elif output == 1:
                    TP += 1
                else:
                    print("wtf")
            else:
                print("wtf")
        accuracies.append((TP + TN) / (TP + TN + FP + FN))
        precisions.append(TP / (TP + FP))
        recalls.append(TP / (TP + FN))
        F1s.append((2.0 * precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1]))
    
    print(f"Congressional Results ({num_trees} trees, {num_folds} folds):")
    print(f"\tAvg Accuracy: {sum(accuracies) / len(accuracies)}")
    print(f"\tAvg Precision: {sum(precisions) / len(precisions)}")
    print(f"\tAvg Recall: {sum(recalls) / len(recalls)}")
    print(f"\tAvg F1 Score: {sum(F1s) / len(F1s)}")

if __name__ == "__main__":
    #main()
    test_wine(50, 5)
    test_congress(50, 5)