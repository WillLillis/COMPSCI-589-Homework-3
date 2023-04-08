#import random_forest
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

def main():
    k_folds = k_folds_gen(10, 'hw3_wine.csv')
    print('hw3_wine.csv')
    for fold in k_folds:
        print(f"New fold: {fold}\n\n")

    k_folds = k_folds_gen(10, 'hw3_house_votes_84.csv')
    print('hw3_house_votes_84.csv')
    for fold in k_folds:
        print(f"New fold: {fold}\n\n")
    
    k_folds = k_folds_gen(10, 'hw3_cancer.csv')
    print('hw3_cancer.csv')
    for fold in k_folds:
        print(f"New fold: {fold}\n\n")


if __name__ == "__main__":
    main()