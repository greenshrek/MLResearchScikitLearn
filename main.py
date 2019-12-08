"""
Author: Pranav Srivastava
Detail: IMDB Score Prediction for a movie
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import datasets
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_selection import f_regression, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn import linear_model
from sklearn import preprocessing

from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC

#univariate feature selection
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

#model selection
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score

#evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

class MLResearchScikitLearn:

    def __init__(self, dataset):
        self.dataset = dataset
        self.nominalfeatures = ['color','director_name','actor_2_name','actor_1_name','actor_3_name','plot_keywords','language','country','content_rating','genres','movie_title']
        self.integerfeatures = ['num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','gross','facenumber_in_poster','num_user_for_reviews','budget','title_year','actor_2_facebook_likes','aspect_ratio','num_voted_users','cast_total_facebook_likes', 'movie_facebook_likes']
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.selected_features = []

    def imputation(self):
        #this function imputes any missing values in the feature column based on the strategy parameter of SimpleImputer class

        #apply imputer with strategy=most_frequent for missing values in string type feature column
        imputer_string = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
        imputer_string.fit(self.dataset[self.nominalfeatures])
        self.dataset[self.nominalfeatures] = imputer_string.transform(self.dataset[self.nominalfeatures])
        
        #apply imputer with strategy=mean for missing values in integer type feature column
        imputer_integer = SimpleImputer(missing_values=np.NaN, strategy="mean")
        imputer_integer.fit(self.dataset[self.integerfeatures])
        self.dataset[self.integerfeatures] = imputer_integer.transform(self.dataset[self.integerfeatures])

    def featureDrop(self, featurename):
        #this function is used to drop any feature based on the request
        self.dataset.drop(featurename,axis=1,inplace=True)

    def featureEncoding(self):
        # This method does the cleaning of feature 'genres'.
        enc= OrdinalEncoder()
        self.dataset[self.nominalfeatures] = enc.fit_transform(self.dataset[self.nominalfeatures])

    def convertFeatureToBinary(self):
        #This method is not used anymore but was used to create a proof of concept for using binary classification
        median = self.dataset["imdb_score"].median()
        for i in range (self.dataset.shape[0]):
            if self.dataset["imdb_score"][i] >= median:
                self.dataset["imdb_score"][i] = 1
            elif self.dataset["imdb_score"][i] < median:
                self.dataset["imdb_score"][i] = 0

    def dataCategorization(self):
        #This function labels the feature column
        self.dataset["imdb_score"]=pd.cut(self.dataset['imdb_score'], bins=[0,4,6,8,10], right=True, labels=False)+1

    def checkOutliers(self):
        #This function draws a graph that shows if there are any outliers
        sns.boxplot(data=pd.DataFrame(self.dataset))
        plt.tight_layout()
        plt.show()

    def handleImbalance(self):
        #This function is used to check if there is any imbalance in the target class
        targets = self.dataset["imdb_score"].value_counts()
        print ("Minority class represents just ",(targets[1]/len(self.dataset["imdb_score"]))*100, " % of the dataset") 

    def confusionMatrix(self,X_train,y_train,X_test,y_test):
        #This function creates a confusion matrix which is a validation technique for the created model.
        #Also, this function prints the accuracy

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred= model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy of model: ", accuracy )

        cf_mat= confusion_matrix(y_true=y_test, y_pred=y_pred)
        print('Confusion matrix of model:\n', cf_mat)


    def univariateFeatureSelection(self):
        #This function selects the top features on the basis of ranking of association between feature and target class

        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score']
        y = self.dataset["imdb_score"]
        selector = SelectKBest(f_classif, k=10)
        #selector = SelectPercentile(F_classif, percentile=25)
        selector.fit(X,y)
        for n,s in zip(self.dataset.columns,selector.scores_):
            print ("Score : ", "for ", n, "is ", s)
        # negate the selector scores to sort them in descending order
        idx = (-selector.scores_).argsort()
        # map index to feature list
        desc_feature = [X.columns[i] for i in idx]
        # select the top 12 feature
        self.selected_features = desc_feature [:12]
        print("top_feature", self.selected_features)


    def treeBaseFeatureSelection(self):
        #this function selects the top features with the help of a decision tree. 
        # Decision tree picks the root node as a feature that provides largest reduction in uncertainity

        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score']
        y = self.dataset["imdb_score"]

        # Build a forest and compute the feature importance
        forest = RandomForestClassifier(n_estimators=250, random_state=0)
        forest.fit(X, y)
        importances = forest.feature_importances_
        print(importances)

        for index in range(X.shape[1]):
            #print(index)
            print ("Importance of feature ", X.columns[index], "is", importances[index])

        # negate the importances to sort them in descending order
        idx = (-importances).argsort()
        # map index to feature list
        desc_feature = [X.columns[i] for i in idx]
        # select the top 12 feature
        self.selected_features = desc_feature [:12]
        print("top_feature", self.selected_features)


    def greedyFeatureSelection(self):
        #this function is the implemenation of greedy backward feature selection

        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score']
        y = self.dataset["imdb_score"]
        decTree = DecisionTreeClassifier()
        scores = model_selection.cross_val_score(decTree, X, y, cv=10)
        print ('Initial Result',scores.mean())

        #Logistic regression is a classification algorithm used to assign observations to a discrete set of classes.
        estimator = linear_model.LogisticRegression(multi_class='auto', solver ='lbfgs')

        #Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
        #cv Determines the cross-validation splitting strategy. 10 denotes 10-fold validation
        rfecv= RFECV(estimator, cv=10)
        rfecv.fit(X, y)

        flag = 0
        for index in rfecv.ranking_:
            if index == 1:
                self.selected_features.append(X.columns[flag])
            flag+=1
        
        print("selected features>", self.selected_features)

    def selectFeatures(self):
        #this function will select the relevant features based on the algorithm and drop the irrelevant features 

        features = self.nominalfeatures + self.integerfeatures
        print("features>>>",features)
        
        #drop the features that were not selected
        self.featureDrop(list(set(features).difference(self.selected_features)))
        print(self.dataset)

    def featureScaling(self):
        #This function normalizes the data across all the columns in the dataset
        #all the columns will have value between 0 and 1

        sc_X = preprocessing.MinMaxScaler()
        self.dataset[self.selected_features] = sc_X.fit_transform(self.dataset[self.selected_features])
        print("dataset______",self.dataset)

    def splitData(self):
        
        split_ds = train_test_split(self.dataset, test_size=int(self.dataset.shape[0]*20/100), random_state = 1)

        train_dataset = split_ds[0]
        test_dataset = split_ds[1]

        self.X_train = train_dataset.loc[:, train_dataset.columns != 'imdb_score'].to_numpy()
        self.y_train = train_dataset["imdb_score"].to_numpy()
        self.X_test = test_dataset.loc[:, test_dataset.columns != 'imdb_score'].to_numpy()
        self.y_test = test_dataset["imdb_score"].to_numpy()

    def kFoldCrossValidation(self):

        allResults= []
        traindata = self.dataset.loc[:, self.dataset.columns != 'imdb_score'].to_numpy()
        target = self.dataset["imdb_score"].to_numpy()

        kf= model_selection.KFold(n_splits=15, shuffle=True, random_state=1)
        
        print("train data  is >>>>>>:")
        clf= RandomForestClassifier(n_estimators = 200)
        #clf = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
        #clf = GaussianNB()
        #clf = SVC(gamma="auto")
        for train_index, test_index in kf.split(traindata):

            #clf= tree.DecisionTreeClassifier()
            clf.fit(traindata[train_index], target[train_index])

            results= clf.predict(traindata[test_index])

            allResults.append(metrics.accuracy_score(results, target[test_index]))

        print ("Accuracy k fold is ", np.mean(allResults))


    def sklearnMetricsRandomForest(self):

        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score'].to_numpy()
        y = self.dataset["imdb_score"].to_numpy()
    
        rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

        param_grid = { 
            'n_estimators': [100, 150, 200, 250, 300],
            'min_samples_split' : np.arange(2, 5),
            'min_samples_leaf' : np.arange(1, 5),
            'max_features': ['auto', 'sqrt', 'log2'],
            'n_jobs' : [-1]
        }

        CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
        CV_rfc.fit(X, y)
        
        print("best parameters for Random Forest Classifier: ",CV_rfc.best_params_)


    def sklearnMetricsDecisionTree(self):
        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score'].to_numpy()
        y = self.dataset["imdb_score"].to_numpy()

        parameters={ 'criterion':['gini','entropy'],
        'max_depth': np.arange(3, 15), 
        'min_samples_leaf':np.arange(1,10), 
        'splitter' : ['best', 'random'], 
        'max_features': ['auto', 'sqrt', 'log2']}

        dtc=tree.DecisionTreeClassifier()
        dtc= GridSearchCV(dtc,parameters, cv=5)
        dtc .fit(X,y)
        
        #dt_model = dtc.best_estimator_
        #print("best model>",dt_model)
        print("Best parameters for Decision Tree Classifier: ", dtc.best_params_)


    def sklearnMetricsKNN(self):

        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score'].to_numpy()
        y = self.dataset["imdb_score"].to_numpy()
    
        pipe_lr= Pipeline([('scl', StandardScaler()), ('clf', KNeighborsClassifier())])
        param_grid= [ {'clf__n_neighbors': list(range(1, 5)),
        'clf__p':[1, 2, 3, 4, 5],
        'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
        'clf__leaf_size': list(range(40, 50)),
        'clf__weights': ['uniform', 'distance']}]
        
        clf= GridSearchCV(pipe_lr, param_grid)
        clf.fit(X, y)
        
        print("\n Best parameters set found on development set:")
        print(clf.best_params_ , "with a score of ", clf.best_score_)


    def createModel(self):

        self.confusionMatrix(self.X_train,self.y_train,self.X_test,self.y_test)

        model = RandomForestClassifier()
        # Train the model using the training sets
        model.fit(self.X_train,self.y_train)

        results = []
        for tf in self.X_test:
            result = ""
            result = model.predict([tf])
            results.append(result[0])

        predicted_results = np.array(results)
        return (metrics.accuracy_score(predicted_results, self.y_test)*100)

dataset = pd.read_csv("movie_metadata.csv")
classobj = MLResearchScikitLearn(dataset)


classobj.imputation()

#as movie_imdb_link is the most irrelevant feature hence dropping it
classobj.featureDrop('movie_imdb_link')
#classobj.checkOutliers()
classobj.featureEncoding()
classobj.dataCategorization()
classobj.handleImbalance()
classobj.treeBaseFeatureSelection()
classobj.selectFeatures()
classobj.featureScaling()

classobj.splitData()
#classobj.kFoldCrossValidation()
classobj.sklearnMetricsKNN()
#classobj.sklearnMetricsRandomForest()
#classobj.sklearnMetricsDecisionTree()
print(classobj.createModel())

#print("nan:>>>>",dataset.isna().any())


