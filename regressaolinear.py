# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:53:03 2023

@author: tiago
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import os

#Importando os dados

df_dados = pd.read_csv("C:\\Projetos\\Python\\RegressãoLinearPython-linkedin\\Advertising.csv")

#Verificando o que foi carregado e a estrutura de dados.

print(df_dados.shape)

#Verificando o conteúdo.
df_dados.head()

#Análise básica da base de dados, utilizando a função describe.
print(df_dados.describe())

#Retirando a coluna que não é necessária, não tem utilidade para a análise.
df_dados.drop(['Unnamed: 0'], axis=1)

#Visualização dos dados de forma gráfica
plt.figure(figsize=(16,8))
plt.scatter(
    df_dados['TV'],
    df_dados['sales'],
    c='red')

plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()

#Criando o modelo para prever o retorno em investimento em TV.

X = df_dados['TV'].values.reshape(-1,1)
y = df_dados['sales'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)

print("O modelo é: Vendas = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

#plotando o modelo

f_previsaoes = reg.predict(X)

plt.figure(figsize = (16,8))

plt.scatter(
    df_dados['TV'],
    df_dados['sales'],
    c='red')
plt.plot(
    df_dados['TV'],
    f_previsaoes,
    c='blue',
    linewidth=3,
    linestyle=':')

plt.xlabel(" ($) Gasto em propaganda de TV")
plt.ylabel(" ($) Vendas")
plt.show()
print('Regressão Linear - Python')





















