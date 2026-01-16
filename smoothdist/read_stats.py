# %%
import sys
import os
import pandas as pd


data_path = "./data"
files_in_path = os.listdir(data_path)
print(f"Files in data path: {files_in_path}")
file_num = 0
file = files_in_path[file_num]

# Load file num
df = pd.read_csv(os.path.join(data_path, file))
# Columns in planning_stats_smooth_distance.csv: 
# ['run_id', 'seed', 'num_obstacles', 'num_path_points', 'ipopt_info', 
# 'total_time', 'ipopt_time', 'success_collision_free', 'min_dist', 
# 'mean_dist', 'p10_dist', 'num_violations']

# Get basic statistics
stats_msg = """
Statistics for file: {}
Number of runs: {}
Success rate: {:.2f}%
Average total time: {:.2f} seconds
Average min distance to obstacles: {:.2f} units
Average mean distance to obstacles: {:.2f} units
Average p10 distance to obstacles: {:.2f} units
Average number of violations: {:.2f} units
-------------------------------
"""
for file in files_in_path:
    df = pd.read_csv(os.path.join(data_path, file))
    stats = (
        file,
        len(df),
        df['success_collision_free'].mean()*100,
        df['total_time'].mean(),
        df['min_dist'].mean(),
        df['mean_dist'].mean(),
        df['p10_dist'].mean(),
        df['num_violations'].mean()
    )
    print(stats_msg.format(*stats))
print("-------------------------------")

# print(f"Statistics for file: {file}")
# print(f"Number of runs: {len(df)}")
# print(f"Success rate: {df['success_collision_free'].mean()*100:.2f}%")
# print(f"Average total time: {df['total_time'].mean():.2f} seconds")
# print(f"Average min distance to obstacles: {df['min_dist'].mean():.2f} units")
# print(f"Average mean distance to obstacles: {df['mean_dist'].mean():.2f} units")
# print(f"Average p10 distance to obstacles: {df['p10_dist'].mean():.2f} units")
# print(f"Average number of violations: {df['num_violations'].mean():.2f}")

