# %%

import pandas as pd
# %%
dados_frutas = 'dados/dados_frutas.xlsx'
df = pd.read_excel(dados_frutas)
df
# %%
from sklearn import tree
arvore = tree.DecisionTreeClassifier()

# %%
#Resposta/O que deve ser previsto
y = df['Fruta']

caracteristicas = ['Arredondada','Suculenta','Vermelha','Doce']
x = df[caracteristicas]
# %%
#ajustar o modelo, vai criar a arvore de decisão
arvore.fit(x, y)


# %%
arvore.predict([[1,1,1,1]])
# %%
arvore.predict([[0,0,0,1]])
# %%
#mostrando a arvore de decisão
import matplotlib.pyplot as plt
plt.figure(dpi=400)
tree.plot_tree(
    arvore,
    feature_names=caracteristicas,
    class_names=arvore.classes_,
    filled=True)

# %%
#mostrando a probabilidade de cada classe dada uma entrada
proba = arvore.predict_proba([[1,1,1,1]])[0]
pd.Series(proba, index=arvore.classes_)