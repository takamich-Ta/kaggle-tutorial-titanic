import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#タイタニックkaggle、データ分析

data=pd.read_csv("./train.csv") #データ読み込み

print(data.isnull().sum()) #nullのデータ確認、数カウント

#女性たくさん生き残る
data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
plt.show() #性別と生存率のグラフ

#等級クラスにも依存する　Pclass:部屋の等級
data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
plt.show() #部屋の等級と生存率のグラフ

#年齢との関係
print(data["Age"].mean())#年齢の平均値
data.loc[data.Age.isnull(),"Age"]=int(data["Age"].mean())
print(data.isnull().sum()) #年齢の平均値でnullを埋めた
data[["Age","Survived"]].groupby(["Age"]).mean().plot.bar()
plt.show() #年齢と生存率のグラフ

#兄弟、夫婦 SibSp:兄弟や夫や妻の人数
data[["SibSp","Survived"]].groupby(["SibSp"]).mean().plot.bar()
plt.show() #兄弟や夫や妻の人数と生存率のグラフ

#親、子供　Parch:親や子供の人数
data[["Parch","Survived"]].groupby(["Parch"]).mean().plot.bar()
plt.show() #親や子供の人数と生存率のグラフ

data.to_csv("./train2.csv") #データ保存
