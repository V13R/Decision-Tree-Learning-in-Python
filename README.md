# Decision Tree Learning

Our project proposes a Machine Learning algorithm, specifically Decision Trees. We have implemented a classification algorithm, using two different methods: entropy classification and Gini impurity respectively. Both entropy and the Gini impurity measure the uncertainty of a classification. The Gini impurity represents the frequency of misclassification. It requires less processing power than computing entropy, because no logarithmic computations are necessary.

The algorithm uses a database from the archive.ics.uci.edu website, where we can find many databases for machine learning purposes. We have chosen 2 databases, with a similar format -the ideal class is on the 0th column, and the classification dimensions are on the following columns-, the "Wine" database, which determines a wine's quality based on 13 factors such as alcohol concentration, color intensity etc. and the "Balance Scale" database, used to predict a scale's position based on the weights standing on each arm and their distance from the center of the scale (4 factors). This data is imported with the importdata function. 70% of it is used to train the decision trees and 30% is used to test them via the split function.

After importing the data and processing it in an appropriate format, two decision trees are trained with the gini_train function and the entropy_train function, respectively. The maximum depth was set to 3 levels, with a condition to have a minimum of 5 objects in each leaf class. This way we avoid both overfitting, in which each object would have its own class, and underfitting, in which classes would be too general to provide necessary information.

The prediction function is used to classify the test data. The accuracy function returns useful information about each tree's accuracy: the confusion matrix (an n*n matrix with the correct labels on the main diagonal, the other values representing misclassifications), the precision and a classification report with other useful information.

Majoritatea funcțiilor au fost implementate cu ajutorul funcțiilor prestabilite din biblioteca sklearn.
La final, datele de ieșire vor fi scrise într-un fișier, output.txt

Most functions were implemented with the help of the sklearn package.
The output is written to the output.txt file.



## References:

https://archive.ics.uci.edu/ml/machine-learning-databases/wine/  	

https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/ 

https://archive.ics.uci.edu/ml/machine-learning-databases/

https://medium.com/machine-learning-101/chapter-3-decision-trees-theory-e7398adac567 

https://medium.com/machine-learning-101/chapter-3-decision-tree-classifier-coding-ae7df4284e99 

http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/ 

https://www.geeksforgeeks.org/decision-tree-implementation-python/ 

https://www.quora.com/What-is-the-interpretation-and-intuitive-explanation-of-Gini-impurity-in-decision-trees/answer/Duane-Rich 

http://www.bel.utcluj.ro/dce/didactic/sisd/SISD_seminar_4_RNA.pdf 

https://scikit-learn.org/

https://scikit-learn.org/stable/modules/tree.html 

https://docs.python.org/3/ 

https://en.wikipedia.org/wiki/F1_score
