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
name = 'wDistnUnc30sx'
folder = '.'
time_df = pd.read_csv(f'{folder}/timeHistory_{name}_{dt}.csv')
w_df = pd.read_csv(f'{folder}/wHistory_{name}_{dt}.csv')
z_df = pd.read_csv(f'{folder}/zHistory_{name}_{dt}.csv')
tau_df = pd.read_csv(f'{folder}/tauHistory_{name}_{dt}.csv')
q_df = pd.read_csv(f'{folder}/qHistory_{name}_{dt}.csv')
qdot_df = pd.read_csv(f'{folder}/qdotHistory_{name}_{dt}.csv')

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

epsilon = 0.5
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
sns.lineplot(data=norm_x_df, x='Time', y='Norm_x', ax=axs[0])
axs[0].axhline(0.432, color='magenta', linestyle='--', label='Bound')
axs[0].set_title('Norm_x')

sns.lineplot(data=norm_w_df, x='Time', y='Norm_w', ax=axs[1])
axs[1].axhline(epsilon, color='r', linestyle='--', label='Epsilon')
# axs[1].axhline(epsilon / 2, color='r')
axs[1].set_title('Norm_w')
axs[0].legend()
axs[1].legend()
# Plot for norm_x_df


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# fig = sns.lineplot(data=norm_w_df, x='Time', y='Norm_w')
# fig.axhline(epsilon, color='r')
# fig.axhline(epsilon / 2, color='r')
# plt.show()
# fig = sns.lineplot(data=norm_x_df, x='Time', y='Norm_x')
# fig.axhline(0.432, color='r')
# plt.show()
# %%
import numpy as np
from scipy.linalg import solve_continuous_are

n=2
A = np.block([[np.zeros((n, n)), np.eye(n)], [np.zeros((n, n)), np.zeros((n, n))]])
B = np.block([[np.zeros((n, n))], [np.eye(n)]])
Q = 2*np.eye(2*n)
R = 5.03*np.eye(n)
P = solve_continuous_are(A, B, Q, R)
radius = 0.432
epsilon = (radius) ** 2 * (np.min(np.linalg.eigvals(Q)) * np.min(np.linalg.eigvals(P)) / np.max(np.linalg.eigvals(P)))
print(epsilon, np.min(np.linalg.eigvals(P)), np.max(np.linalg.eigvals(P)))

# %%
