import numpy as np
import csv
import plotly.graph_objects as go
import antropy as ant


PATH = './processed/anomaly_scores.csv'

y = []
x = []

with open(PATH, newline='') as f:
    reader = csv.reader(f)
    y = list(reader)
    x = list(range(len(y)))
# print(y)
y = [e[0] for e in y]
entropy = ant.app_entropy(np.array(y).astype(np.float))
fig = go.Figure()
fig.add_annotation(
            text='ENtropy_value: (%d)'%entropy,
            showarrow=True,
            arrowhead=1)
config = dict({'scrollZoom': True})

fig.add_trace(
    go.Scatter(
        x=x,
        y=y))

fig.show(config=config)