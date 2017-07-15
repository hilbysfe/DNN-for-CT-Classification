import numpy as np
from sklearn.manifold import TSNE

X = np.load('./features/l2norm.npy')
labels = np.load('./features/labels.npy')

model = TSNE(n_components = 2, random_state=0)

Y = model.fit_transform(X);
np.save('./features/l2norm_tsne_1000.npy', Y)