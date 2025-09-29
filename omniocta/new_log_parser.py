import re
import pickle
import numpy as np
import scipy as sp
import uaibot as ub
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from uaibot_cpp_bind import expSO3, SmapSO3, SmapSE3, expSE3, ECdistance
import plotly.colors as pc
from plotly.subplots import make_subplots


def vector_field_plot(
    coordinates,
    field_values,
    orientations,
    curve,
    num_arrows=10,
    init_ball=0,
    final_ball=None,
    num_balls=10,
    add_lineplot=False,
    colorscale=None,
    show_curve=True,
    ball_size=5,
    curve_width=2,
    path_width=5,
    frame_scale=0.05,
    frame_width=2,
    curr_path_style="solid",
    prev_path_style="dash",
    **kwargs,
):
    """Plot a vector field in 3D. The vectors are represented as cones and the
    auxiliary lineplot is used to represent arrow tails. The kwargs are passed
    to the go.Cone function. Also plots the target curve, and the path of the
    object. The object is represented as a sphere. The orientations are represented
    as frames with the x, y and z axis of the frame.

    Parameters
    ----------
    coordinates : list or np.array
        Mx3 array of coordinates of the vectors. Each row corresponds to x,y,z
        respectively. The column entries are the respective coordinates.
    field_values : list or np.array
        Mx3 array of field values of the vectors. Each row corresponds to u,v,w
        respectively, i.e. the LINEAR velocity of the field in each direction.
        The column entries are the respective values.
    orientations : list or np.array
        Mx3x3 array of orientations of the object. Each row corresponds to the
        orientation of the object at that point. The 'column' entries are the
        respective 3x3 rotation matrices.
    curve : np.array
        Nx3 array of the curve points. Each row corresponds to x,y,z respectively.
    num_arrows : int, optional
        Number of vector field arrows (cones) to plot. The default is 10.
    init_ball : int, optional
        Initial ball index to plot. The default is 0.
    final_ball : int, optional
        Final ball index to plot. The default is None, which plots until the end.
    num_balls : int, optional
        Number of balls to plot. The default is 10.
    add_lineplot : bool, optional
        Whether to add a lineplot of the field coordinates. The default is False.
        This is used to connect the vector field arrows.
    colorscale : list, optional
        List of colors to use in the plot. The default is None, which uses the
        Plotly default colors. The list must have at least 6 colors, which are
        used for the curve, previous path, current path, initial ball, final ball
        and the object, respectively.
    show_curve : bool, optional
        Whether to show the target curve. The default is True.
    ball_size : int, optional
        Size of the object balls. The default is 5.
    curve_width : int, optional
        Width of the curve line. The default is 2.
    path_width : int, optional
        Width of the path line. The default is 5.
    frame_scale : float or list, optional
        Scale factor for the orientation frames. The default is 0.05. If a list
        is given, the scale factor is applied to each axis of the frame.
    frame_width : int, optional
        Width of the orientation frame lines. The default is 2.
    curr_path_style : str, optional
        Style of the current path line. The default is "solid".
    prev_path_style : str, optional
        Style of the previous path line. The default is "dash".
    **kwargs
        Additional keyword arguments to pass to the go.Cone function.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Resulting plotly figure.
    """
    if final_ball is None:
        final_ball = len(coordinates) - 1

    if isinstance(frame_scale, (int, float)):
        frame_scale = [frame_scale] * 3

    coordinates = np.array(coordinates).reshape(-1, 3)
    arrows_idx = np.round(np.linspace(0, len(coordinates) - 1, num_arrows)).astype(int)
    coord_field = coordinates[arrows_idx].T
    field_values = np.array(field_values).reshape(-1, 3)[arrows_idx].T
    ball_idx = np.round(np.linspace(init_ball, final_ball, num_balls)).astype(int)
    coord_balls = coordinates[ball_idx]
    ori_balls = np.array(orientations)[ball_idx]
    coordinates = coordinates.T

    if colorscale is None:
        colorscale = pc.qualitative.Plotly

    if isinstance(curve, tuple):
        curve = curve[0]

    fig = go.Figure()

    # Curve
    if show_curve:
        fig.add_trace(
            go.Scatter3d(
                x=curve[:, 0],
                y=curve[:, 1],
                z=curve[:, 2],
                mode="lines",
                line=dict(width=curve_width, color=colorscale[1]),
            )
        )
    # Previous path
    if init_ball > 0:
        fig.add_trace(
            (
                go.Scatter3d(
                    x=coordinates[0, 0:init_ball],
                    y=coordinates[1, 0:init_ball],
                    z=coordinates[2, 0:init_ball],
                    mode="lines",
                    line=dict(
                        width=path_width, dash=prev_path_style, color=colorscale[5]
                    ),
                )
            )
        )

    # Current path
    fig.add_trace(
        go.Scatter3d(
            x=coordinates[0, init_ball:final_ball],
            y=coordinates[1, init_ball:final_ball],
            z=coordinates[2, init_ball:final_ball],
            mode="lines",
            line=dict(width=path_width, dash=curr_path_style, color=colorscale[0]),
        )
    )

    # Vector field arrows
    fig.add_trace(
        go.Cone(
            x=coord_field[0, :],
            y=coord_field[1, :],
            z=coord_field[2, :],
            u=field_values[0, :],
            v=field_values[1, :],
            w=field_values[2, :],
            colorscale=[[0, colorscale[5]], [1, colorscale[5]]],  # Set the colorscale
            showscale=False,
            **kwargs,
        )
    )

    # Orientation frames
    if orientations is not None:
        for i, ori in enumerate(ori_balls):
            px, py, pz = coord_balls[i, :]
            ux, uy, uz = ori[:, 0] / (np.linalg.norm(ori[:, 0] + 1e-6)) * frame_scale
            vx, vy, vz = ori[:, 1] / (np.linalg.norm(ori[:, 1] + 1e-6)) * frame_scale
            wx, wy, wz = ori[:, 2] / (np.linalg.norm(ori[:, 2] + 1e-6)) * frame_scale
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + ux],
                    y=[py, py + uy],
                    z=[pz, pz + uz],
                    mode="lines",
                    line=dict(color="red", width=frame_width),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + vx],
                    y=[py, py + vy],
                    z=[pz, pz + vz],
                    mode="lines",
                    line=dict(color="lime", width=frame_width),
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[px, px + wx],
                    y=[py, py + wy],
                    z=[pz, pz + wz],
                    mode="lines",
                    line=dict(color="blue", width=frame_width),
                )
            )

    # Object
    for i, coord in enumerate(coord_balls):
        if i == 0:
            color = colorscale[3]
        elif i == len(coord_balls) - 1:
            color = colorscale[4]
        else:
            color = "rgba(172, 99, 250, 0.6)"
        fig.add_trace(
            go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode="markers",
                marker=dict(size=ball_size, color=color),
            )
        )

    if add_lineplot:
        fig.add_scatter3d(
            x=coordinates[0, :], y=coordinates[1, :], z=coordinates[2, :], mode="lines"
        )

    return fig



def progress_bar(i, imax):
    """Prints a progress bar in the terminal."""
    bar_len = 60
    filled_len = int(round(bar_len * i / float(imax)))

    percents = round(100.0 * i / float(imax), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    print(f'[{bar}] {percents}%\r', end='')
    if i == imax:
        print()

def pose2htm(p, R):
    """Homogeneous transformation matrix from position and rotation."""
    p = np.array(p)
    htm = np.eye(4)
    htm[0:3, 0:3] = R
    htm[0:3, 3] = p.ravel()
    return htm

def get_stable_index(distances, threshold=0.7, window_size=30):
    """Get the index where the distance to the curve is stable, i.e. the
    average of the last 30 samples is below the threshold.
    """
    for i in range(len(distances) - window_size):
        if np.mean(distances[i:i+window_size]) < threshold:
            return i
    return -1

def check_traversal(indexes, n_points):
    """Check if the system traversed the whole curve. Returns True if the
    system traversed the whole curve, False otherwise. Also returns the
    time spent in seconds, assuming a control frequency of 100 Hz.
    Accepts 98% of the points to account for noise.
    """
    indexes = np.array(indexes)
    unique_indexes = np.unique(indexes)
    success = len(unique_indexes) >= 0.98 * n_points
    # Checks the time spent
    dt = 10e-3
    time_spent = len(indexes) * dt
    return success, time_spent

#%%

path = "/home/fbartelt/Projects/robotics-experiments/omniocta/data/"
# Get all pickle files in the path
files = [f for f in os.listdir(path) if f.endswith('.pkl')]
pos_std_opts = sorted(list(set(re.findall(r"pos_(\d+\.\d+)_", "\n".join(files)))))
ori_std_opts = sorted(list(set(re.findall(r"ori_(\d+\.\d+)_", "\n".join(files)))))
all_combinations = [(p, ori_std_opts[i]) for i, p in enumerate(pos_std_opts)]

stats_by_pair = {}
for i, (pos_std, ori_std) in enumerate(all_combinations):
    progress_bar(i, len(all_combinations))
    pair_files = [f for f in files if f.startswith(f"pos_{pos_std}_ori_{ori_std}")]
    pos_means, pos_stds, pos_mins, pos_maxs = [], [], [], []
    ori_means, ori_stds, ori_mins, ori_maxs = [], [], [], []
    dist_means, dist_stds, dist_mins, dist_maxs = [], [], [], []
    time_traversal = []
    fails_convergence, fails_traversal = 0, 0
    for file in pair_files:
        with open(os.path.join(path, file), 'rb') as file_:
            data = pickle.load(file_)
        pos_err = np.array(data["true_pos_error"]) * 100
        ori_err = np.rad2deg(np.array(data["true_ori_error"]))
        true_distance = data["true_distance"]
        stable_index = get_stable_index(true_distance, threshold=0.7, window_size=30)
        if stable_index == -1:
            print(f"File {file} did not stabilize.")
            fails_convergence += 1
        else:
            pos_means.append(np.mean(pos_err[stable_index:]))
            pos_stds.append(np.std(pos_err[stable_index:]))
            pos_mins.append(np.min(pos_err[stable_index:]))
            pos_maxs.append(np.max(pos_err[stable_index:]))
            ori_means.append(np.mean(ori_err[stable_index:]))
            ori_stds.append(np.std(ori_err[stable_index:]))
            ori_mins.append(np.min(ori_err[stable_index:]))
            ori_maxs.append(np.max(ori_err[stable_index:]))
            dist_means.append(np.mean(true_distance[stable_index:]))
            dist_stds.append(np.std(true_distance[stable_index:]))
            dist_mins.append(np.min(true_distance[stable_index:]))
            dist_maxs.append(np.max(true_distance[stable_index:]))
            nearest_indexes = data["true_nearest_indexes"][stable_index:]
            success_flag, time_spent = check_traversal(nearest_indexes, 500)
            success = data["traversed"][-1]
            if not success:
                print(f"File {file} did not traverse the whole curve.")
                fails_traversal += 1

            else:
                time_traversal.append(time_spent)

    stats_by_pair[f"{pos_std}_{ori_std}"] = {
        "mean_avg_pos_err": np.mean(pos_means),
        "mean_std_pos_err": np.mean(pos_stds),
        "mean_min_pos_err": np.mean(pos_mins),
        "mean_max_pos_err": np.mean(pos_maxs),
        "std_avg_pos_err": np.std(pos_means),
        "std_std_pos_err": np.std(pos_stds),
        "std_min_pos_err": np.std(pos_mins),
        "std_max_pos_err": np.std(pos_maxs),
        "mean_avg_ori_err": np.mean(ori_means),
        "mean_std_ori_err": np.mean(ori_stds),
        "mean_min_ori_err": np.mean(ori_mins),
        "mean_max_ori_err": np.mean(ori_maxs),
        "std_avg_ori_err": np.std(ori_means),
        "std_std_ori_err": np.std(ori_stds),
        "std_min_ori_err": np.std(ori_mins),
        "std_max_ori_err": np.std(ori_maxs),
        "mean_avg_dist": np.mean(dist_means),
        "mean_std_dist": np.mean(dist_stds),
        "mean_min_dist": np.mean(dist_mins),
        "mean_max_dist": np.mean(dist_maxs),
        "mean_time_traversal": np.mean(time_traversal),
        "std_time_traversal": np.std(time_traversal),
        "min_time_traversal": np.min(time_traversal) if time_traversal else None,
        "max_time_traversal": np.max(time_traversal) if time_traversal else None,
        "all_avg_pos_errs": pos_means,
        "all_avg_ori_errs": ori_means,
        "all_avg_dist": dist_means,
        "fails_convergence": fails_convergence,
        "fails_traversal": fails_traversal,
        }

with open(os.path.join(path, 'summary_stats2.pkl'), 'wb') as file:
    pickle.dump(stats_by_pair, file)

print("Done processing all files.")

# %%
df = pd.DataFrame.from_dict(stats_by_pair)

ix = 8
df[[pos_std_opts[ix] + '_' + ori_std_opts[ix]]]

# df[["pos_std", "ori_std", "mean_avg_pos_err", "std_avg_pos_err"]]

# Split into position df and orientation df
# pos_df = df.T[["mean_avg_pos_err", "std_avg_pos_err", "mean_min_pos_err", "mean_max_pos_err", "fails"]]
# pos_df
# ori_df = df.T[["mean_avg_ori_err", "std_avg_ori_err", "mean_min_ori_err", "mean_max_ori_err", "fails"]]
# ori_df

for val in ("pos", "ori"):
    print(f"Results for {val} error:")
    result_table = pd.DataFrame(index=ori_std_opts, columns=pos_std_opts)

    # Fill the result table
    for pos_std in pos_std_opts:
        for ori_std in ori_std_opts:
            col_name = f"{pos_std}_{ori_std}"
            value = df[col_name].loc[f'mean_avg_{val}_err']
            result_table.loc[ori_std, pos_std] = np.round(value, 2)

    result_table.index = result_table.index.astype(float).round(2)
    result_table.columns = result_table.columns.astype(float).round(2)
    print(result_table)

#%%
# Get subdf with all columns and only "all_avg_pos_errs" and "all_avg_ori_errs"
pos_df = df.T[["all_avg_pos_errs"]]
ori_df = df.T[["all_avg_ori_errs"]]
# Create boxplot using plotly
fig = px.box(pos_df, points="all", y="all_avg_pos_errs", title="Position error distribution for all parameter combinations", labels={"index": "Parameter combination (pos_std, ori_std)", "all_avg_pos_errs": "Position error (cm)"})
fig.show()

#%%
import plotly.express as px
import plotly.io as pio
import pandas as pd
import webbrowser
import tempfile
import os

def plot_heatmap_in_browser(dataframe, title="Mean Avg Pos Error"):
    """
    Plots a heatmap using Plotly and opens it in the default web browser.

    Parameters:
    - dataframe: A pandas DataFrame with numeric row and column labels and values.
    - title: Title for the plot.
    """
    # Reset index to get ori_std as a column for plotly
    df_plot = dataframe.copy()
    df_plot['ori_std'] = df_plot.index
    df_melted = df_plot.melt(id_vars='ori_std', var_name='pos_std', value_name='mean_avg_pos_err')

    # Create heatmap
    fig = px.density_heatmap(
        df_melted,
        x="pos_std",
        y="ori_std",
        z="mean_avg_pos_err",
        color_continuous_scale="Viridis",
        text_auto=True,
        title=title
    )

    # Improve layout
    fig.update_layout(
        xaxis_title="pos_std",
        yaxis_title="ori_std",
        title_x=0.5,
        width=1200,
        height=800,
    )

    # Save to a temporary HTML file and open in browser
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        tmp_path = tmpfile.name
        pio.write_html(fig, file=tmp_path, auto_open=False)
        webbrowser.open("file://" + os.path.realpath(tmp_path))

plot_heatmap_in_browser(result_table, title="Mean Avg Pos Error")
#%%
# Get files per combination (pos_std, ori_std)

# for f in files:
#     with open(os.path.join(path, f), 'rb') as file:
#         pos_std = re.findall(r'pos_(\d+\.\d+)', f)
#         ori_std = re.findall(r'ori_(\d+\.\d+)', f)
#         print(f"Processing file: {f} with pos_std: {pos_std} and ori_std: {ori_std}")
#         data = pickle.load(file)
#         p_hist = data['p_hist']
#         R_hist = data['R_hist']
#
#         avg_dist, avg_pos_err, avg_ori_err = get_average_stable_errors(p_hist, R_hist, curve)
#         stats[f"{pos_std[0]}_{ori_std[0]}"] = {
#             'avg_dist': avg_dist,
#             'avg_pos_err': avg_pos_err,
#             'avg_ori_err': avg_ori_err,
#         }
# Create df and order by index
df = pd.DataFrame.from_dict(stats, orient='index')
df
# Plot distances of each file
# fig = go.Figure()
# for i, dist in enumerate(average_dists):
#     fig.add_trace(go.Scatter(y=dist, mode='lines+markers', name=f'Run {i+1}'))
# fig.show()
# print(average_dists)

path = "/home/fbartelt/Projects/robotics-experiments/omniocta/data/gibberish.pkl"
with open(path, 'rb') as file:
    results = pickle.load(file)

df = pd.DataFrame.from_dict(results)
print(df.columns)
sub_df = df[["pos_std", "ori_std", "mean_avg_pos_err",  "std_avg_pos_err", "mean_avg_ori_err", "std_avg_ori_err"]]
# sub_df.sort_values(by=["mean_avg_pos_err", "std_avg_pos_err"], )
sub_df.sort_values(by=["pos_std", "ori_std"], )

np.array(df[["all_avg_pos_errs"]].iloc[-1].values[0]).max()
#%%
path = "/home/fbartelt/Projects/robotics-experiments/omniocta/data/pos_0.2222222222222222_ori_0.011111111111111112_seed_0.pkl"

with open(path, 'rb') as file:
    data = pickle.load(file)

p_hist = data['p_hist']
R_hist = data['R_hist']
v_hist = data['v_hist']
n_points = 5000
r, b, d = 2.5, 1, 0.2
curve = precomputed_hd(hd, n_points, r, b, d)
curve_pos = np.array([c[:3, 3] for c in curve])
print(data.keys())
pos_err = data["true_pos_error"]
np.mean(pos_err[int(len(pos_err) * 0.4):]*100)
np.std(pos_err[int(len(pos_err) * 0.4):]*100)
np.max(pos_err[int(len(pos_err) * 0.4):]*100)
np.min(pos_err[int(len(pos_err) * 0.4):]*100)
np.median(pos_err[int(len(pos_err) * 0.4):]*100)
sp.stats.kurtosis(pos_err[int(len(pos_err) * 0.4):]*100)
# np.mean(df.iloc[0]['all_avg_pos_errs'])
fig = vector_field_plot(p_hist, v_hist, R_hist, curve_pos, num_arrows=0, init_ball=0, final_ball=len(p_hist)-1, num_balls=20, add_lineplot=False, show_curve=True, ball_size=3, frame_scale=0.1)
fig.show()
# go.Figure(go.Scatter(y=data['dist_hist'].ravel(), mode='lines')).show()



#%%
mean_dist, mean_pos, mean_ori, dist_hist, pos_err_hist, ori_err_hist = get_average_stable_errors(p_hist, R_hist, curve)

def nvim_err_plot(dist_hist, pos_err_hist, ori_err_hist):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Distance to curve", "Position error (cm)", "Orientation error (deg)"))
    xvec = np.arange(len(dist_hist)) * 10e-3
    fig.add_trace(
        go.Scatter(
            y=dist_hist,
            mode='lines',
            name='Distance to curve',
            line=dict(color='blue'),
        )
    , row=1, col=1)
    fig.add_trace(
        go.Scatter(
            y=pos_err_hist,
            mode='lines',
            name='Position error (cm)',
            line=dict(color='orange'),
        )
    , row=2, col=1)
    fig.add_trace(
        go.Scatter(
            y=ori_err_hist,
            mode='lines',
            name='Orientation error (deg)',
            line=dict(color='green'),
        )
    , row=3, col=1)

    return fig

fig = nvim_err_plot(dist_hist, pos_err_hist, ori_err_hist)
fig.show()


# %%
path = "/home/fbartelt/Projects/robotics-experiments/omniocta/data/"
# Get all pickle files in the path
files = [f for f in os.listdir(path) if f.endswith('.pkl')]
pos_std_opts = sorted(list(set(re.findall(r"pos_(\d+\.\d+)_", "\n".join(files)))))
ori_std_opts = sorted(list(set(re.findall(r"ori_(\d+\.\d+)_", "\n".join(files)))))
all_combinations = [(p, ori_std_opts[i]) for i, p in enumerate(pos_std_opts)]
all_combinations = sorted(all_combinations, key=lambda x: (float(x[0]), float(x[1])))
print(all_combinations)

combination_num = 8
pos_std, ori_std = all_combinations[combination_num]
pair_files = [f for f in files if f.startswith(f"pos_{pos_std}_ori_{ori_std}")]
print(pair_files)

for file in pair_files:
    with open(os.path.join(path, file), 'rb') as file_:
        data = pickle.load(file_)
        if any(data['traversed']):
            print(f"File {file} traversed the curve.")


print("finish")
print(data.keys())

