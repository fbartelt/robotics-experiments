#%%
import sys
sys.path.append('./build')
import pybind_cruzancona
# %%
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dt = '1e6'
name = 'wDistnUnc2'
time_df = pd.read_csv(f'timeHistory_{name}_{dt}.csv')
w_df = pd.read_csv(f'wHistory_{name}_{dt}.csv')
z_df = pd.read_csv(f'zHistory_{name}_{dt}.csv')
q_df = pd.read_csv(f'qHistory_{name}_{dt}.csv')
qdot_df = pd.read_csv(f'qdotHistory_{name}_{dt}.csv')

# For quantized data
unique_indices = time_df.drop_duplicates().index
norm_w = np.linalg.norm(w_df.loc[unique_indices].values, axis=1)
norm_w_df = pd.DataFrame({'Time': time_df.loc[unique_indices]['Time'], 'Norm_w': norm_w})
norm_x = z_df.loc[unique_indices].iloc[:, :4]
norm_x_df = pd.DataFrame({'Time': time_df.loc[unique_indices]['Time'], 'Norm_x': np.linalg.norm(norm_x.values, axis=1)})

# For full data
# norm_w = np.linalg.norm(w_df[::4].values, axis=1)
# norm_w_df = pd.DataFrame({'Time': time_df[::4]['Time'], 'Norm_w': norm_w})
# norm_x = z_df[::4].iloc[:, :4]
# norm_x_df = pd.DataFrame({'Time': time_df[::4]['Time'], 'Norm_x': np.linalg.norm(norm_x.values, axis=1)})
fig = sns.lineplot(data=norm_w_df, x='Time', y='Norm_w')
fig.axhline(0.05, color='r')
plt.show()
# %%
