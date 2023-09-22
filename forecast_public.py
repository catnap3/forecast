import pandas as pd
import numpy as np

data = pd.read_csv("気象データ1.csv", encoding="Shift-JIS")

#print(data.head())

from sklearn.linear_model import LinearRegression as lr #線形回帰のライブラリ
#from sklearn.linear_model import LogisticRegression as LR #ロジスティック回帰
#from sklearn.naive_bayes import GaussianNB # ナイーブベイズ_正規分布(ガウシアン分布)
#from sklearn.naive_bayes import BernoulliNB # ナイーブベイズ_ベルヌーイ分布
from sklearn.model_selection import train_test_split #全データを訓練用と評価用にどのくらいの配分で分けるかを指示するためのライブラリ

X_data = data[['平均気温(℃)',
               '天気概況(昼:06時~18時)',
               #'日照時間(時間)',
               '平均風速(m/s)',
               #'平均蒸気圧(hPa)',
               '平均湿度(%)',
               '平均雲量(10分比)']] #独立変数/特徴量
Y_data = data['降水量(0or1)'] #従属変数/ターゲット。雨が少しでも降っていれば1, 降水量が0の場合のみ0。

scores = []
for i in range(10000):
    X_train, X_test, y_train, y_test = train_test_split(X_data,Y_data,test_size=0.2)
    clf = lr()
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    scores.append(clf.score(X_test,y_test))
print(np.median(scores))