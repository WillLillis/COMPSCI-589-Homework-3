from copy import deepcopy
import decision_tree
import misc

class random_forest:
    # Add various tree params as forest params...
    def __init__(self, data: list, num_trees: int):
        # list containing all trees that are members of the forest
        self.trees = []
        # helper dictionary to translate between category labels and (column) indices
        self.cat_to_attr_index = {}
        for index in range(len(data[0])):
            self.cat_to_attr_index[data[0][index]] = index

        for _ in range(num_trees):
            tree_data = misc.bootstrap(data)
            self.trees.append(decision_tree.decision_tree(deepcopy(tree_data), None, '', is_root=True, split_metric="Gini"))

    def classify_instance(self, instance: list):
        votes = {}
        # collect the votes from all of the member trees
        for worker in self.trees: 
            vote = worker.classify_instance(instance, self.cat_to_attr_index)
            if vote in votes:
                votes[vote] += 1
            else:
                vote[votes] = 1
        # and return the classification with the most votes
        return max(votes, key = votes.get)
