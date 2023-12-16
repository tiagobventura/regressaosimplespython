# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:10:04 2023

@author: tiago

Regressão linear simples, com uso apenas de biblioteca para desenhar o gráfico.
"""
import matplotlib.pyplot as plt
import pandas as pd

df_dados = pd.read_csv("C:\\Projetos\\Python\\RegressãoLinearPython-linkedin\\Advertising.csv")

"""
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
"""
x = df_dados['TV']
y = df_dados['sales']

#Função para calcular a média.
def calcular_media(lista):
    return sum(lista)/len(lista)

#Função para calcular os coeficientes da regressão linear
def regressao_linear_simples(x, y):
    n = len(x)
    
    media_x = calcular_media(x)
    media_y = calcular_media(y)
    
    #Calcula os coeficeintes da regressão.
    numerador = sum((xi - media_x) * (yi - media_y) for xi, yi in zip(x, y))
    denominador = sum((xi - media_x)**2 for xi in x)
    
    #Coefiente angular(slope)
    beta1 = numerador/denominador
    
    #Coeficiente linear (intercept)
    beta0 = media_y - beta1 * media_x
    
    return beta0, beta1

#Funçaõ para fazer previsões com os coeficientes da regressão
def prever(regressao, x):
    beta0, beta1 = regressao
    return beta0 + beta1 * x


#Calcula os coeficientes da regressão.
regressao = regressao_linear_simples(x, y)

#Faz uma precsão para um novo valor de x
novo_valor_x = 150
previsao = prever(regressao, novo_valor_x)

#Gera pontos para o gráfico da linha de regresão.
linha_x = [min(x), max(x)]
linha_y = [prever(regressao, xi) for xi in linha_x]

#Exibe os resultados.
print("Coeficiente Linear (Intercept):", regressao[0])
print("Coeficiente Angular (Slope):", regressao[1])
print(f"Previsão para x={novo_valor_x}: {previsao}")

#Cria gráfico 
plt.scatter(x, y, label='Dados')
plt.plot(linha_x, linha_y, color='red', label='Regressão Linear')
plt.scatter(novo_valor_x, previsao, color='green', label=f'Previsãopara x={novo_valor_x}')

#Adiciona rótulos e legenda
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Mostra o gráfico
plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

