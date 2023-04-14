from data_processing import get_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# get data
df = get_data()
print(df.describe())
print(df.info())

with pd.option_context('display.max_rows', 10,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(df)

df.hist(bins=50, figsize=(10, 5))
plt.show()

df["score_cat"] = pd.cut(df["review/score"], bins=[0., 1.0, 2.0, 3.0, 4.0, 5.0], labels=[1, 2, 3, 4, 5])
df["score_cat"].value_counts().sort_index().plot(kind="bar")
plt.show()

#
# df["score_cat"].hist()
# plt.show()
