from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

#kaggleタイタニック、学習して精度確認

data=pd.read_csv("./train3.csv")#加工済みデータ読み込み
data.drop(["Unnamed: 0"],axis=1,inplace=True)#保存の際に入ったやつ削除

#学習用に変換
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']

model=svm.SVC(kernel="rbf",C=1,gamma=0.1) #SVM
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print("Accuracy(rbf SVM):",metrics.accuracy_score(prediction,test_Y)) #精度確認
