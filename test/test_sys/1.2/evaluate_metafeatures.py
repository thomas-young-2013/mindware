from automlToolkit.components.meta_learning.meta_features import calculate_all_metafeatures
import numpy as np

np.random.seed(1)
X = np.random.rand(100, 5)
y = np.array([np.random.randint(5) for _ in range(100)])
meta = calculate_all_metafeatures(X=X,
                                  y=y,
                                  categorical=[False] * 5,  # Categorical mask, list of bool
                                  dataset_name="default")
print(meta.load_values())

# from autosklearn.pipeline.implementations.OneHotEncoder import OneHotEncoder
#
# pred1 = OneHotEncoder().fit_transform(X).toarray()
# print(pred1)
#
# from sklearn.preprocessing import OneHotEncoder
#
# pred2 = OneHotEncoder(categorical_features=[False] * 5).fit_transform(X)
# print(pred2)
