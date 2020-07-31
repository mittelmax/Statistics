import pandas as pd
import missingno as msn
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)

#Reading the dataframe
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df = pd.concat([df_train, df_test])

#Checking variable types
df.info()

#Transforming some variables into categorys
to_cat = ["Survived", "Pclass", "Sex", "Name", "Embarked"]
df[to_cat] = df[to_cat].astype("category")
df.info()

#Checking for NA's
msn.matrix(df)

#Majority of Cabin column is missing so it will be removed
#Removing Ticket and Cabin columns
df = df.drop(columns = ["Ticket", "Cabin"])

#Substituting missing ages with mean of the ages
ages_mean = np.mean(df.Age)
df.Age = df.Age.fillna(ages_mean)
print(df.head(20))

#Investigating treatment pronouns
print(df.Name)
pronouns = ["Mr.", "Miss.", "Master.", "Mrs.", "Don.", "Rev.", "Mme.", "Major", "Dr.", "Col."]
pronouns_vec = []

#Looping over repeated pronouns
for i in df.Name:
    for j in pronouns:
        if j in i:
            pronouns_vec.append(j)

pronouns_vec = pd.Series(pronouns_vec)

#Replacing name column with pronouns vector
df.Name = pronouns_vec
df.columns = df.columns.str.replace("Name", "Pronoun")

#TConverting column to categorical
df.Pronoun = df.Pronoun.astype("category")

#Checking for any missing value left
msn.matrix(df)

#Checking Embark factors
df.Embarked.value_counts()

#Replacing NA's with the mode
print(df.Embarked.isna().sum())
print(df.Fare.isna().sum())
df.Embarked = df.Embarked.fillna("S")
fare_mean = np.mean(df.Fare)
df.Fare = df.Fare.fillna(fare_mean)

#Dividing training and test set

df.reset_index(inplace=True)

df_train = df.loc[:890, "PassengerId":]
df_test = df.loc[891:, "PassengerId":]
df_test = df_test.drop(columns = ["Survived"])

#Applying models
X = df_train.loc[:, "Pclass":"Embarked"]
X_test = df_test.loc[:, "Pclass":"Embarked"]
y = df_train["Survived"]


#Feature Scaling numerical values
scaler = MinMaxScaler(feature_range=(0, 1))
X.loc[:, "Age":"Fare"] = scaler.fit_transform(X.loc[:, "Age":"Fare"])
X_test.loc[:, "Age":"Fare"] = scaler.fit_transform(X_test.loc[:, "Age":"Fare"])


# Creating the encoder
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X)
encoder.fit(X_test)

# Applying the encoder
X = encoder.transform(X)
X_test = encoder.transform(X_test)

#Using Kfold cv to evaluate models

#Perceptron
perc = Perceptron(tol=1e-3, random_state=0)
perc.fit(X, y)
score_perceptron = np.mean(cross_val_score(perc, X, y, cv=10))

#Logistic Regression
logi = LogisticRegression()
logi.fit(X, y)
score_logi = np.mean(cross_val_score(logi, X, y, cv=10))

#SVM
supportVec = svm.SVC()
supportVec.fit(X, y)
score_svm = np.mean(cross_val_score(supportVec, X, y, cv=10))

#Gradient Boosting Machine
gbm = GradientBoostingClassifier()
gbm.fit(X, y)
gbm_score = np.mean(cross_val_score(gbm, X, y, cv=10))

#Random Forest
rdf = RandomForestClassifier()
rdf.fit(X, y)
rdf_score = np.mean(cross_val_score(rdf, X, y, cv=10))


#Visualing model scores
models = ["Perceptron", "Logistic Regression", "SVM", "GBM", "rdf"]
scores = [score_perceptron, score_logi, score_svm, gbm_score, rdf_score]
graph = sns.barplot(models, scores)

#Prediction on test set
#Gbm was used due to best performance
df_test.reset_index(inplace=True)
predictions = pd.DataFrame(gbm.predict(X_test).astype(int))
id = pd.DataFrame(df_test.PassengerId)

#Preparing data for Kaggle
predictions = id.join(predictions)
predictions.rename(columns={0: "Survived"}, inplace=True)

#Exporting csv
predictions.to_csv("prediction.csv", index=False)
