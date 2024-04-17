import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from proplot import rc
import pdb

# Recreate the DataFrame with correct data types for error bars
data = {
    'Task': [
        # 'window-close', 'window-open', 'reach', 'push', 'drawer-close',
        'Average'
    ],
    'Flat R3M': [13.33],
    'Lang Diffuser': [16.67],
    'SkillDiffuser': [23.33]
}

error = {
    'SkillDiffuser': [4.7, 6.3, 9.5, 3.8, 2.1]
}

# Convert data and error to DataFrame
df = pd.DataFrame(data)
df_error = pd.DataFrame(error)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(3, 5))

ax.set_xlim(-0.4, 0.6)


# Colors for each bar group
# colors = ['green', 'blue', 'yellow', 'lightcoral', 'grey', 'purple']
colors = [
    # '#8c564b',  # Brown color for "Random"

    # '#7f7f7f',  # Gray color for "Random"
# '#8c564b',  # Brown color for "Random"
    '#4292E3',  # Blue color for "Pixel"
# '#038972',  # Green color for "LCRL"
    '#F2B974',  # Yellow color for "LISA"
# '#98df8a',  # Light green

# '#ff7f0e',  # Orange color
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
    if i == 0:
        offset = 0.03
    elif i == 1:
        offset = 0
    elif i == 2:
        offset = -0.03
    ax.bar(bar_positions + i * width, df[column], width=width, label=column, color=color, yerr=yerr, capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Task Instruction', family="Times New Roman", fontsize=12, weight='bold')
ax.set_ylabel('Success Rate (%)', family="Times New Roman", fontsize=12, weight='bold')
# ax.set_title('Success Rates by Task Instruction and Method', fontsize=14, weight='bold')
ax.set_xticks(bar_positions + width)
ax.set_xticklabels(df['Task'], rotation=0, ha="center")
# ax.legend()

# Make the y-axis label, ticks and tick labels match the line color.
ax.tick_params(axis='y', which='major', labelsize=10)
ax.tick_params(axis='x', which='major', labelsize=12)

# plt.rcParams['font.sans-serif'] = ['Arial']
# Setting the style to be suitable for CVPR publication
plt.style.use('seaborn-whitegrid')
plt.rcParams['font.family'] = 'TeX Gyre Schola'
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rc('font',family='Times New Roman')

# Save the plot to a file suitable for inclusion in a CVPR paper.
plt.savefig('./CVPR_metaworld_bar_chart2.png', format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
