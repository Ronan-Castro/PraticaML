# %%

import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
# %%
dados_cerveja = 'dados/dados_clones.parquet'
#reading analytical base table
df = pd.read_parquet(dados_cerveja)
df.columns
# %%
features = ['Massa(em kilos)', 'Estatura(cm)', 'Tempo de existência(em meses)']
target = 'Status '
X = df[features]
y = df[target]
#X = X.replace({
#    "Tipo ": "",
#    "Yoda": 0,
#    "Shaak Ti": 1
#    })
# %%
df['General Jedi encarregado'].unique()
# %%
model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)
# %%
tree.plot_tree(
        model,
        feature_names=features,
        class_names=model.classes_,
        filled=True,
        max_depth=4)
# %%
