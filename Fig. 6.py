import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

data_dir = './results'
dot_color = '#6AC2AF'
line_color = '#ED7D31'
font_name = 'Times New Roman'
font_size = 38
legend_font_size = 34
tick_font_size = 38
marker_size = 38
rated_capacity = 155

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.linewidth'] = 1.2

file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy') and '256' in f]

for file_name in file_list:
    model_name = file_name.replace('_prediction_results.npy', '')
    file_path = os.path.join(data_dir, file_name)
    data = np.load(file_path, allow_pickle=True).item()

    soh_true = np.array(data["soh_targets"])
    soh_pred = np.array(data["soh_predictions"])

    rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
    rmse_percent = (rmse / rated_capacity) * 100
    r2 = r2_score(soh_true, soh_pred)
    mape = mean_absolute_percentage_error(soh_true, soh_pred) * 100

    print(f"Model: {model_name.upper()}")
    print(f"  RMSE: {rmse:.4f} Ah ({rmse_percent:.2f}%)")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print("-" * 30)

    plt.figure(figsize=(8, 7))
    plt.scatter(soh_true, soh_pred, c=dot_color, s=50, label="Estimated")
    plt.plot([soh_true.min(), soh_true.max()],
             [soh_true.min(), soh_true.max()],
             linestyle='--', color=line_color,  linewidth=3.5)

    plt.xlabel("True capacity", fontsize=font_size)
    plt.ylabel("Estimated capacity", fontsize=font_size)

    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)

    ax = plt.gca()

    xticks = np.arange(110, 155, 10)
    yticks = np.arange(110, 155, 10)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    plt.tight_layout()
    plt.show()