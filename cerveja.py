# %%

import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
# %%
dados_cerveja = 'dados/dados_cerveja.xlsx'
df = pd.read_excel(dados_cerveja)
df.head()
# %%
features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'
X = df[features]
y = df[target]
X = X.replace({
    "mud": 1, "pint": 2,
    "sim": 1, "não": 0,
    "clara": 1, "escura":0
    })
X
# %%
model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)
# %%
tree.plot_tree(
        model,
        feature_names=features,
        class_names=model.classes_,
        filled=True)
# %%
