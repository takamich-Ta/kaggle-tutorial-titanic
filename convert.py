import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#タイタニックkaggle、データ加工

data=pd.read_csv("./train2.csv") #データ読み込み

data["Age_band"]=0 #年齢データ変換、年齢で0から4に分ける
data.loc[data["Age"]<=16,"Age_band"]=0 #Ageの代わりに用いる
data.loc[(data["Age"]>16)&(data["Age"]<=32),"Age_band"]=1
data.loc[(data["Age"]>33)&(data["Age"]<=48),"Age_band"]=2
data.loc[(data["Age"]>48)&(data["Age"]<=64),"Age_band"]=3
data.loc[data["Age"]>64,"Age_band"]=4

data[["Age_band","Survived"]].groupby(["Age_band"]).mean().plot.bar()
plt.show() #年齢層と生存率のグラフ

#数字データにする
data["Sex"].replace(["male","female"],[0,1],inplace=True)

#いらないデータ削除
data.drop(["Name","Ticket","Fare","Cabin","PassengerId","Embarked","Age"],axis=1,inplace=True)

data.to_csv("./train3.csv") #データ保存
