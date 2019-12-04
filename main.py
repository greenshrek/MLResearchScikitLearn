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
            print(self.dataset["imdb_score"][5])
            if self.dataset["imdb_score"][i] >= median:
                self.dataset["imdb_score"][i] = 1
            elif self.dataset["imdb_score"][i] < median:
                self.dataset["imdb_score"][i] = 0

    def checkOutliers(self):
        sns.boxplot(data=pd.DataFrame(self.dataset["imdb_score"]))
        plt.show()

    def featureSelection(self):
        X = self.dataset.loc[:, self.dataset.columns != 'imdb_score']
        print(X)
        y = self.dataset["imdb_score"]
        print(y)
        # Build a forest and compute the feature importance
        forest = RandomForestClassifier(n_estimators=250, random_state=0)
        forest.fit(X, y)
        importances = forest.feature_importances_
        print("importante: ",importances)
        for index in range(X.shape[1]):
            #print(index)
            print ("Importance of feature ", index, "is", importances[index])

dataset = pd.read_csv("movie_metadata.csv")

classobj = MLResearchScikitLearn(dataset)

classobj.imputation()
classobj.featureEncoding()


classobj.convertFeatureToBinary()
classobj.featureSelection()
#classobj.checkOutliers()
print(dataset)
#print("nan:>>>>",dataset.isna().any())
