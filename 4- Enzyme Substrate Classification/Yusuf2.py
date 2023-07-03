''' Sınıflandırma ve Regresyon'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sbn
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier, BaggingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

''' Regresyon'''
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

''' Clustering '''

from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans




def boslari_doldur_siniflandirma(veri, Hedef):

    ''' -İlk fonksiyonumuz veri üzerinde genel bir işlem yapar ve Sınıflandırma Algoritmalarını kullanır'''
    veri.columns = veri.columns.str.replace(' ', '_').str.lower()
    
    for col in veri.columns:
        if 'id' in col or 'name' in col:
            veri.drop(col, axis=1, inplace=True)

    for sutun in veri.columns:
        if veri[sutun].dtype == 'O':
            veri[sutun] = veri[sutun].fillna(veri[sutun].mode()[0])
        else:
            veri[sutun] = veri[sutun].fillna(veri[sutun].mean())

    bool_sutun = veri.columns[(veri.nunique() <= 2) & (veri.dtypes == bool)]
    veri[bool_sutun] = veri[bool_sutun].astype(int)

    atilacak_sozel_sutun = veri.columns[(veri.nunique() > 15) & (veri.dtypes == object)]
    veri.drop(atilacak_sozel_sutun, axis = 1, inplace = True)

    x = veri.drop(Hedef, axis = 1)
    y = veri[Hedef]
    x = pd.get_dummies(x, drop_first = True)

    def siniflandirma(x, y):
        model1 = LogisticRegression()
        model2 = DecisionTreeClassifier()
        model3 = RandomForestClassifier()
        model4 = BernoulliNB()
        model5= GaussianNB()
        model6 = SVC()
        model7 = GradientBoostingClassifier()
        model8 = AdaBoostClassifier()
        model9 = HistGradientBoostingClassifier()
        model10 = BaggingClassifier()
        model11 = XGBClassifier()
    
        models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11]
        model_names = ['Logistic', 'Decision', 'Random', 'Bernoulli', 'Gaussian', 'Support', 'Gradient',
                    'AdaBoost', 'Hist', 'Bagging', 'XGBoost']
        
        x_train, x_test, y_train, y_true = train_test_split(x, y, test_size = 0.2, random_state = 42)
        x_train = StandardScaler().fit_transform(x_train)
        x_test = StandardScaler().fit_transform(x_test)

        acc = []
        skor = pd.DataFrame(columns = ['Accuracy'], index = model_names)
        
        for model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc.append(accuracy_score(y_true, y_pred) * 100)
            
        skor['Accuracy'] = acc

        print(skor.sort_values('Accuracy', ascending = False))
        

        #return skor.sort_values('Accuracy', ascending = False)
        plt.figure(figsize=(10, 6))
        #sbn.countplot(data=skor, y = 'Accuracy', x = skor.index, color='green')
        sbn.barplot(data = skor, y ='Accuracy', x = skor.index, color = 'lightblue')
        for i, v in enumerate(skor['Accuracy']):
            plt.text(i, v, str(round(v, 2)), color='black', ha='center')
        plt.xlabel('Accuracy')
        plt.ylabel('Models')
        plt.title('Accuracy Scores of Classification Models')
        plt.show()


    
    return siniflandirma(x, y)

#######################################################################################
## Regresyon algoritması

def boslari_doldur_regresyon(veri, Hedef):

    ''' -İkinci fonksiyonumuz veri üzerinde genel bir işlem yapar ve Regresyon Algoritmalarını kullanır'''
    veri.columns = veri.columns.str.replace(' ', '_').str.lower()
    
    for col in veri.columns:
        if 'id' in col or 'name' in col:
            veri.drop(col, axis=1, inplace=True)

    for sutun in veri.columns:
        if veri[sutun].dtype == 'O':
            veri[sutun] = veri[sutun].fillna(veri[sutun].mode()[0])
        else:
            veri[sutun] = veri[sutun].fillna(veri[sutun].mean())

    bool_sutun = veri.columns[(veri.nunique() <= 2) & (veri.dtypes == bool)]
    veri[bool_sutun] = veri[bool_sutun].astype(int)

    atilacak_sozel_sutun = veri.columns[(veri.nunique() > 15) & (veri.dtypes == object)]
    veri.drop(atilacak_sozel_sutun, axis = 1, inplace = True)

    x = veri.drop(Hedef, axis = 1)
    y = veri[Hedef]
    x = pd.get_dummies(x, drop_first = True)

    def regresyon(x, y):
        lin = LinearRegression()
        rid = Ridge()
        las = Lasso()
        ela = ElasticNet()
        sup = SVR(kernel = "sigmoid")
        ran = RandomForestRegressor()
        dec = DecisionTreeRegressor()
        
        models = [lin, rid, las, ela, sup, ran, dec]
        model_names = ["Linear_R", "Ridge", "Lasso", "Elastic", "Support", "Random", "Decision_T"]
        
        x_train, x_test, y_train, y_true = train_test_split(x, y, test_size = 0.2, random_state = 42)
        x_train = MinMaxScaler().fit_transform(x_train)
        x_test = MinMaxScaler().fit_transform(x_test)
        
        r2 = []
        
        skor = pd.DataFrame(columns = ["R2_Score"], index = model_names)
        
        for model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            r2.append(r2_score(y_true, y_pred) * 100)
        
            
        skor["R2_Score"] = r2

        return skor.sort_values('R2_Score', ascending = False)


    return regresyon(x, y)

    


def boslari_doldur_kümeleme(veri):

    ''' -İlk fonksiyonumuz veri üzerinde genel bir işlem yapar ve Sınıflandırma Algoritmalarını kullanır'''
    veri.columns = veri.columns.str.replace(' ', '_').str.lower()
    
    for col in veri.columns:
        if 'id' in col or 'name' in col:
            veri.drop(col, axis=1, inplace=True)

    for sutun in veri.columns:
        if veri[sutun].dtype == 'O':
            veri[sutun] = veri[sutun].fillna(veri[sutun].mode()[0])
        else:
            veri[sutun] = veri[sutun].fillna(veri[sutun].mean())

    bool_sutun = veri.columns[(veri.nunique() <= 2) & (veri.dtypes == bool)]
    veri[bool_sutun] = veri[bool_sutun].astype(int)

    atilacak_sozel_sutun = veri.columns[(veri.dtypes == object)]
    veri.drop(atilacak_sozel_sutun, axis = 1, inplace = True)

    # x = veri.drop(Hedef, axis = 1)
    # y = veri[Hedef]
    # x = pd.get_dummies(x, drop_first = True)
    pca = PCA(n_components = 2)
    x = pca.fit_transform(veri)
    x = pd.DataFrame(x, columns = ["PCA1", "PCA2"])


    def kumeleme(x):
        model = KMeans(4)
        model = model.fit(x)
        tahmin = model.predict(x)
        x['kume'] = tahmin
        plt.figure(figsize = (10, 6))
        plt.scatter(x["PCA1"], x["PCA2"], c = tahmin, cmap="viridis")
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.title("Clustering")
        plt.colorbar()
        plt.show()


        fig = px.scatter_3d(x, x = 'PCA1', y = 'PCA2', z = 'kume',
              color='kume')
        fig.show()

    
    return kumeleme(x)




def sadece_boslari_doldur(veri):

    veri.columns = veri.columns.str.replace(' ', '_').str.lower()
    
    for col in veri.columns:
        if 'id' in col or 'name' in col:
            veri.drop(col, axis=1, inplace=True)

    for sutun in veri.columns:
        if veri[sutun].dtype == 'O':
            veri[sutun] = veri[sutun].fillna(veri[sutun].mode()[0])
        else:
            veri[sutun] = veri[sutun].fillna(veri[sutun].mean())

    bool_sutun = veri.columns[(veri.nunique() <= 2) & (veri.dtypes == bool)]
    veri[bool_sutun] = veri[bool_sutun].astype(int)

    atilacak_sozel_sutun = veri.columns[(veri.nunique() > 15) & (veri.dtypes == object)]
    veri.drop(atilacak_sozel_sutun, axis = 1, inplace = True)
