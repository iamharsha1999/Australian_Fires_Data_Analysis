import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_modis = pd.read_csv('Dataset/CSV_3.csv')

times_m = df_modis.groupby(['acq_date'])['acq_date'].count().index.tolist()
frames_data_m = [df_modis.loc[df_modis['acq_date'] == t] for t in times_m]

for t in frames_data_m:
    t.confidence = le.fit_transform(t.confidence)

## MODIS Data
fig = go.Figure(data = go.Densitymapbox(lat=frames_data_m[0].latitude, lon=frames_data_m[0].longitude, z=frames_data_m[0].confidence, radius=3, colorscale='Cividis'))
fig.update_layout(mapbox_style="white-bg",
        mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ],mapbox_center_lon=135, mapbox_center_lat=-28.34, mapbox_zoom=3.75, title = times_m[0])
fig.show()
