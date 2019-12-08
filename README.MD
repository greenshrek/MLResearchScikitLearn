# Environment
    - Use python3.7

# Files
- main.py : Main file where which has all the implementation.

- imdb_multiclassification_model.sav : This is the model that was created and saved during the study of this project

- movie_metadata.csv : IMDB dataset

# Execution
- you can simply do a run using 
    ```
    python main.py

    ```

- main.py has a class which has functions. Every fucntion is related to the step in the machine learning lifecycle. Many of the functions are commented as they have been used in the study and the results are recorded in the report. However, you can explore any function by uncommenting it. Below is the list of functions.
    ```
#classobj.checkOutliers()
classobj.featureEncoding()
classobj.dataCategorization()
classobj.handleImbalance()


#classobj.univariateFeatureSelection()
classobj.treeBaseFeatureSelection()
#classobj.greedyFeatureSelection()


classobj.selectFeatures()
#classobj.heatMap()
classobj.featureScaling()

classobj.splitData()
#classobj.kFoldCrossValidation()
#classobj.sklearnMetricsKNN()
#classobj.sklearnMetricsRandomForest()
#classobj.sklearnMetricsDecisionTree()
#lassobj.sklearnMetricSVM()
classobj.createModel()

#use SVC, 
#classobj.evaluationMetrics("SVC")

classobj.generategraph()

    ```