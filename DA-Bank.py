# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:26:31 2022

@author: tyoma
"""

import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='whitegrid')
from PIL import Image

#Titres

st.title('Analyse des campagnes promotionnelles bancaires')
st.caption('par Karim SABER-CHERIF et Artem VALIULIN')
image = Image.open(r"C:\Users\tyoma\Downloads\header.png")
st.image(image, width=800)
st.markdown('Le jeu de données initial :')

#chargement du jeu de données

df = pd.read_csv(r'C:\bank.csv')
st.dataframe(df)

st.title("Phase de l'analyse")

#poutcome
st.markdown('La réponse en fonction de la campagne précédente :')
st.image(Image.open(r"C:\Users\tyoma\Downloads\poutcome.png"))


#job
st.markdown('La réponse en fonction du métier du client :')
st.image(Image.open(r"C:\Users\tyoma\Downloads\job.png"))

#balance
st.markdown('La réponse en fonction des moyens sur le compte bancaire :')
st.image(Image.open(r"C:\Users\tyoma\Downloads\balance.png"))

#pearson
st.markdown('La heatmap du test de Pearson :')
st.image(Image.open(r"C:\Users\tyoma\Downloads\pearson.png"))

#features
st.markdown('Les variables séléctionnées par les deux modèles :')
st.image(Image.open(r"C:\Users\tyoma\Downloads\tree.png"))
st.image(Image.open(r"C:\Users\tyoma\Downloads\boosting.png"))

#courbe
st.markdown('La courbe ROC :')
st.image(Image.open(r"C:\Users\tyoma\Downloads\courbe.png"))

st.title("Phase de modélisation")

df_new = df.copy() # On enregistre df dans df_new pour ne pas corrompre les données de df
df_new.deposit = df_new.deposit.replace(['no', 'yes'], [0, 1])
df_new = df_new.drop(['duration'], axis=1)

data = df_new.drop('deposit', axis=1)
target = df_new['deposit']

df_new['campaign'] = df_new['campaign'].apply(lambda x : df_new.campaign.mean() if x > 35 else x)

# On remplace les variables binaires par 0 ou 1
data.housing = data.housing.replace(['no', 'yes'], [0, 1])
data.loan = data.loan.replace(['no', 'yes'], [0, 1])
data.default = data.default.replace(['no', 'yes'], [0, 1])

# On crée des variables indicatrices à partir des variables catégorielles
data = data.join(pd.get_dummies(data.marital, prefix='marital'))
data = data.join(pd.get_dummies(data.job, prefix='job'))
data = data.join(pd.get_dummies(data.contact, prefix='contact'))
data = data.join(pd.get_dummies(data.month, prefix='month'))
data = data.join(pd.get_dummies(data.poutcome, prefix='pout'))
data = data.join(pd.get_dummies(data.education, prefix='edu'))

  
# On supprimes les variables dont on a plus besoin
#data = data.drop(['balance', 'marital', 'job', 'contact', 'age', 'month', 'poutcome', 'pdays', 'education'], axis=1)
data = data.drop(['marital', 'job', 'contact', 'month', 'poutcome', 'education'], axis=1)

# TEST DES MODELE EN STANDARDISANT LES VARIABLES NUMERIQUES
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# on split nos données
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

# On standardise nos variables quantitatives
var = ['age', 'balance', 'campaign', 'previous', 'age']
scaler = StandardScaler()
X_train[var] = scaler.fit_transform(X_train[var])
X_test[var] = scaler.transform(X_test[var])


model = st.selectbox(label="Choix du modèle", options=["Gradient Boosting", "Decision Tree"])

def get_model(model):
    if model == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators = 200,
                                  learning_rate=0.1,
                                  max_depth = 6,
                                  random_state = 234)
    elif model == "Decision Tree":
        model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=234)
    model.fit(X_train, y_train)
    score = model.score(X_test,y_test)
    return score

st.write("Score test :", get_model(model))