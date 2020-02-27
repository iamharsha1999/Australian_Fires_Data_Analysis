import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go


df_modis = pd.read_csv('Dataset/CSV_3.csv')

times_m = df_modis.groupby(['acq_date'])['acq_date'].count().index.tolist()
frames_data_m = [df_modis.loc[df_modis['acq_date'] == t] for t in times_m]

print(frames_data_m[0].latitude)
## MODIS Data
fig = go.Figure(data = go.Densitymapbox(lat=frames_data_m[0].latitude, lon=frames_data_m[0].longitude, z=frames_data_m[0].frp, radius=6, colorscale='Hot'))
fig.update_layout(mapbox_style = 'open-street-map',mapbox_center_lon=135, mapbox_center_lat=-25.34)
fig.show()
