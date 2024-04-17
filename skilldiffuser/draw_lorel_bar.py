import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from proplot import rc
import pdb

# Recreate the DataFrame with correct data types for error bars
data = {
    'Task': [
        'close drawer', 'open drawer', 'turn faucet left', 'turn faucet right',
        'move black mug right', 'move white mug down', 'Average'
    ],
    'Random': [52, 14, 24, 15, 12, 5, 20],
    'LCBC': [50, 0, 12, 31, 73, 6, 29],
    'LCRL': [58, 8, 13, 0, 0, 0, 13],
    'Lang DT': [10, 60, 0, 0, 20, 0, 15],
    'LISA': [100, 20, 0, 30, 60, 30, 40],
    'SkillDiffuser': [95, 55, 55, 25, 18, 10, 43]
}
error = {
    'SkillDiffuser': [3.2, 13.3, 9.3, 4.4, 3.9, 1.7, 1.1]
}

# Convert data and error to DataFrame
df = pd.DataFrame(data)
df_error = pd.DataFrame(error)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(12, 6))

# Colors for each bar group
# colors = ['green', 'blue', 'yellow', 'lightcoral', 'grey', 'purple']
colors = [
    # '#8c564b',  # Brown color for "Random"

    '#7f7f7f',  # Gray color for "Random"
    # '#1f77b4',  # Blue color for "Pixel"
'#4292E3',  # Blue color for "Pixel"
'#038972',  # Green color for "LCRL"
    # '#F2B974',  # Yellow color for "LISA"
'#8ED799',  # Light green

# '#ff7f0e',  # Orange color
'#F2B974',  # Yellow color for "LISA"
    # '#8c564b',  # Brown color for "Random"
    # '#d62728',  # Red color for "Ours"
'#D78189',  # Red color for "Ours"

]


# The x position of bars
bar_positions = np.arange(len(df['Task']))

# Bar width
width = 0.12

# Plotting each bar group
for i, (column, color) in enumerate(zip(df.columns[1:], colors)):
    # yerr = df_error[column] if column in df_error else None
    yerr = None
    ax.bar(bar_positions + i * width, df[column], width, label=column, color=color, yerr=yerr, capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Task Instruction', family="Times New Roman", fontsize=14, weight='bold')
ax.set_ylabel('Success Rate (%)', family="Times New Roman", fontsize=14, weight='bold')
# ax.set_title('Success Rates by Task Instruction and Method', fontsize=14, weight='bold')
ax.set_xticks(bar_positions + width * 2.5)
ax.set_xticklabels(df['Task'], rotation=15, ha="right", fontsize=12)
ax.legend()

# Make the y-axis label, ticks and tick labels match the line color.
ax.tick_params(axis='y', which='major', labelsize=12)
ax.tick_params(axis='x', which='major', labelsize=12)

# plt.rcParams['font.sans-serif'] = ['Arial']
# Setting the style to be suitable for CVPR publication
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'TeX Gyre Schola'
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rc('font',family='Times New Roman')

# Save the plot to a file suitable for inclusion in a CVPR paper.
plt.savefig('./CVPR_style_bar_chart.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
