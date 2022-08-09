import pandas as pd
import numpy as np
from numpy import random
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (20,5)

import warnings
warnings.filterwarnings(action='ignore')

df60s = pd.read_csv("dataset-of-60s.csv")
df70s = pd.read_csv("dataset-of-70s.csv")
df80s = pd.read_csv("dataset-of-80s.csv")
df90s = pd.read_csv("dataset-of-90s.csv")
df00s = pd.read_csv("dataset-of-00s.csv")
df10s = pd.read_csv("dataset-of-10s.csv")

#print(df60s)

#print(df60s.columns)

#target is whether the song is a hit or not 1 = hit song, 0 = not a hit

#this is the full df
#df = pd.concat(map(pd.read_csv, ["dataset-of-60s.csv", "dataset-of-70s.csv", "dataset-of-80s.csv", "dataset-of-90s.csv", "dataset-of-00s.csv", "dataset-of-10s.csv"]), ignore_index = True)
#df.dropna()

##print(df)

#print(df.iloc[-1:])

#print(df.tail())

#checking if df is properly filled ^^

#print(df.info())

dfs = [pd.read_csv(f"dataset-of-{decade}0s.csv") for decade in ['6','7','8','9','0','1']]
print(dfs[1])
for i, decade in enumerate([1960,1970,1980,1990,2000,2010]):
    dfs[i]['decade'] = pd.Series(decade, index = dfs[i].index)

print(dfs[5])



#shuffle our complete df of all decades so that they are not in order

df = pd.concat(dfs, axis = 0).sample(frac = 1.0, random_state = 1).reset_index(drop = True)
print(df)

#check na and drop object because not relevant

print(df.info())


#make a copy for editing and preprocessing
def preprocess(df):
    dfCopy = df.copy()

    #we want to drop categorical values that have nothing to do with our analysis, track name, artist name, and uri (link from spotify api)
    #there are too many elements in these columns we would have to set up too many dummy variables thus making a df with too many cols
    dfCopy = dfCopy.drop(["track", "artist", "uri"], axis = 1)

    #since we predict target (hit or not) we split it

    y = dfCopy["target"]
    x = dfCopy.drop("target", axis = 1)

    #training and testing
    #higher training % = more accuracy, common practice is to use 70/30
    # due to size of our dataset (small) we will use 80/20
    x_train,  x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state = 1)

    #scale values to make them closer together

    scale = StandardScaler()
    scale.fit(x_train)
    x_train = pd.DataFrame(scale.transform(x_train), index = x_train.index, columns = x_train.columns)
    x_test = pd.DataFrame(scale.transform(x_test), index = x_test.index, columns = x_test.columns)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = preprocess(df)

print(x_train.var()) #var close to 1
print(x_train.mean()) #mean close to 0

#print(df.var()) variance is too high, must scale in preprocess

#training data

mlModels = {
    "                   Logistic Regression": LogisticRegression(),
    "                   K-Nearest Neighbors": KNeighborsClassifier(),
    "                         Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine (Linear Kernel)": LinearSVC(),
    "   Support Vector Machine (RBF Kernel)": SVC(),
    "                        Neural Network": MLPClassifier()
}


#training individual models
#for name, model in mlModels.items():
    #model.fit(x_train, y_train)
    #print(name + " trained.")

#results of accuracy

#for name, model in mlModels.items():
    #print(name + ": {:.2f}%".format(model.score(x_test, y_test) * 100))


#after determining the score, we can see that SVM for RBF kernel and Neural Network are the most precise algs
#therefore, we will procede using those

svcModel = SVC()
NNmodel = MLPClassifier()

svcModel.fit(x_train, y_train)
NNmodel.fit(x_train, y_train)

y1_pred = svcModel.predict(x_test)
y2_pred = NNmodel.predict(x_test)

#calculate a confusion matrix

cnf_matrix = metrics.confusion_matrix(y_test, y1_pred)
cnf2_matrix = metrics.confusion_matrix(y_test, y2_pred)

#creating the map
mpl.rcParams['figure.figsize']=(10,5)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="RdPu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#method for generating songs for prediction

def make_song(danceability = x_train["danceability"].mean(),
                energy = x_train["energy"].mean(),
                key = x_train["key"].mean(),
                loudness = x_train["loudness"].mean(),
                mode = x_train["mode"].mean(),
                speechiness = x_train["speechiness"].mean(),
                acousticness = x_train["acousticness"].mean(),
                instrumentalness = x_train["instrumentalness"].mean(),
                liveness = x_train["liveness"].mean(),
                valence = x_train["valence"].mean(),
                tempo = x_train["tempo"].mean(),
                duration_ms = x_train["duration_ms"].mean(),
                time_signature = x_train["time_signature"].mean(),
                chorus_hit = x_train["chorus_hit"].mean(),
                sections = x_train["sections"].mean()):
                decade = 2010
                return pd.DataFrame({   "danceability": danceability,
                                        "energy": energy,
                                        "key": key,
                                        "loudness": loudness,
                                        "mode": mode,
                                        "speechiness": speechiness,
                                        "acousticness": acousticness,
                                        "instrumentalness": instrumentalness,
                                        "liveness": liveness,
                                        "valence": valence,
                                        "tempo": tempo,
                                        "duration_ms": duration_ms,
                                        "time_signature": time_signature,
                                        "chorus_hit": chorus_hit,
                                        "sections": sections,
                                        "decade": decade
                }, index = [0])

#the average song is a flop "0"
print(svcModel.predict(make_song()))

#now lets look at some important features of the dataset to inquire what to predict
df = pd.DataFrame(x_train)

#here we will test multiple samples of the df ie. different songs and see which ones are hits and if they fit in our ranges
pd.set_option('display.max_columns', None)
for i in range(0,10):
    sample = df.sample()
    
    print(sample)
    print(svcModel.predict(sample))

for i in range(0,10):
    sample = df.sample()
    print(sample)
    print(NNmodel.predict(sample))



