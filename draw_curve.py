import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./coordinates_annotated.csv')
scores_data = data.groupby('year')['score'].mean()
years = scores_data.index.values
scores = scores_data.to_numpy()
fig = plt.figure(figsize=(8, 5))
plt.plot(years, scores, marker='o')
plt.xlabel('years')
plt.ylabel('overlapped area ratio')
plt.title('Overlapped area ratio w.r.t. the year of the map')
plt.savefig('curve.png')
