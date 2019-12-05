import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier

#univariate feature selection
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

#mode selection
from sklearn import tree
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics

class MLResearchScikitLearn:

    def __init__(self, dataset):
        self.dataset = dataset
        self.nominalfeatures = ['color','director_name','actor_2_name','actor_1_name','actor_3_name','plot_keywords','language','country','content_rating','genres','movie_title','movie_imdb_link']
        self.integerfeatures = ['num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','gross','facenumber_in_poster','num_user_for_reviews','budget','title_year','actor_2_facebook_likes','aspect_ratio']

    def imputation(self):

        #apply imputer with strategy=most_frequent for missing values in string type feature column
        imputer_string = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
        imputer_string.fit(self.dataset[self.nominalfeatures])
        self.dataset[self.nominalfeatures] = imputer_string.transform(self.dataset[self.nominalfeatures])
        
        #apply imputer with strategy=mean for missing values in integer type feature column
        imputer_integer = SimpleImputer(missing_values=np.NaN, strategy="mean")
        imputer_integer.fit(self.dataset[self.integerfeatures])
        self.dataset[self.integerfeatures] = imputer_integer.transform(self.dataset[self.integerfeatures])

    def featureEncoding(self):
        enc= OrdinalEncoder()
        self.dataset[self.nominalfeatures] = enc.fit_transform(self.dataset[self.nominalfeatures])

    def convertFeatureToBinary(self):
        median = self.dataset["imdb_score"].median()
        for i in range (self.dataset.shape[0]):
            if self.dataset["imdb_score"][i] >= median:
                self.dataset["imdb_score"][i] = 1
            elif self.dataset["imdb_score"][i] < median:
                self.dataset["imdb_score"][i] = 0

    def checkOutliers(self):
        sns.boxplot(data=pd.DataFrame(self.dataset["imdb_score"]))
        plt.show()

    def handleImbalance(self):
        targets = self.dataset["imdb_score"].value_counts()
        #print (targets)
        print ("Minority class represents just ",(targets[1]/len(self.dataset["imdb_score"]))*100, " % of the dataset") 

    def univariateFeatureSelection(self):
        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score']
        y = self.dataset["imdb_score"]
        selector = SelectPercentile(f_regression, percentile=25)
        selector.fit(X,y)
        for s in selector.scores_:
            print ("Score : ", s)

    def treeBaseFeatureSelection(self):
        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score']
        y = self.dataset["imdb_score"]
        # Build a forest and compute the feature importance
        forest = RandomForestClassifier(n_estimators=250, random_state=0)
        forest.fit(X, y)
        importances = forest.feature_importances_
        for index in range(X.shape[1]):
            #print(index)
            print ("Importance of feature ", index, "is", importances[index])

    def kFoldCrossValidation(self):
        allResults= []
        split_ds = train_test_split(self.dataset, test_size=int(self.dataset.shape[0]*20/100), random_state = 1)

        train = split_ds[0]
        test = split_ds[1]

        train_data = train.loc[:, train.columns != 'imdb_score']
        train_data = train_data.to_numpy()

        train_target = train["imdb_score"]
        train_target = train_target.to_numpy()  

        kf= model_selection.KFold(n_splits=6, shuffle=True, random_state=1)
        
        print("train data  is >>>>>>:")

        for train_index, test_index in kf.split(train_data):
            clf= tree.DecisionTreeClassifier()
            clf.fit(train_data[train_index], train_target[train_index])

            results= clf.predict(train_data[test_index])

            allResults.append(metrics.accuracy_score(results, train_target[test_index]))

        print ("Accuracy is ", np.mean(allResults))

    def kFoldTest(self):
        allResults= []
        iris = datasets.load_iris()
        print(iris)


dataset = pd.read_csv("movie_metadata.csv")
print("initial dataset", dataset)
classobj = MLResearchScikitLearn(dataset)
classobj.imputation()
classobj.featureEncoding()
classobj.convertFeatureToBinary()
#classobj.handleImbalance()
#classobj.univariateFeatureSelection()
#classobj.checkOutliers()
classobj.kFoldCrossValidation()

#print("nan:>>>>",dataset.isna().any())


