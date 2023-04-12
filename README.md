My code can be run via the command ``python3 src.py``. The program takes no additional command line arguments.

The code's dependencies are:

- the ``deepcopy`` function from the ``copy`` module

- the ``csv`` module

- the ``randrange`` function from the ``random`` module

- the ``sqrt`` and ``log2`` functions from the ``math`` module

- the ``random`` module

Each run of the program as it is currently configured completes a k-folds cross validation test of both the wine and congressional datasets, with ``k=5`` folds. 
The number of folds and number of trees in the random forest can be configured via the arguments passed to ``test_wine()`` and ``test_congress()`` inside of ``src.py``.

The splitting and stopping criteria can be configured by altering the corresponding arguments passed the ``decision_tree`` constructor within the ``random_forest`` constructor.