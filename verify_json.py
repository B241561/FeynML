import pandas as pd
import plotly.express as px
import json

df = pd.DataFrame({'A': [1, 2, 3]})
fig = px.histogram(df, x='A')
print("JSON Length:", len(fig.to_json()))
