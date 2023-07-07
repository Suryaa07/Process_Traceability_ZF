import pandas as pd
import seaborn as sns
import matplotlib .pyplot as plt
import numpy as np

dt=pd.read_csv("racprod.csv")

dt.head()

dt.columns

dt.describe()

dt.info

corr=dt.corr()
corr

input_var=dt['Trace Date']
v1=dt['Brake sw. bracket Torque \n(Screw 1)']
v2=dt['Brake sw. bracket Torque \n(Screw 2)']
v3=dt['Clutch sw. bracket Torque (Screw 1)']
v4=dt['Clutch sw. bracket Torque (Screw 2)']
v5=dt['DBV torque (Bolt 1)']
v6=dt['DBV torque (Bolt 2)']
v7=dt['DBV torque (Bolt 3)']
v8=dt['DBV torque (Bolt 4)']
v9=dt['CMC Torque (Bolt 1)']
v10=dt['CMC Torque (Bolt 2)']
v11=dt['Pedal shaft_Assy_Torq']
v12=dt['Pre_Play_mm']
v13=dt['Brake Pedal_Crackoff_mm']
v14=dt['Brake Pedal']
v15=dt['Brake Pedal_Fullstr_mm']
v16=dt['Brake Switch']
v17=dt['Brake Pedal HS']
v18=dt['ClutchPedal FreePlay']
v19=dt['Clutch Pedal FullStroke']
v20=dt['Clutch Pedal HS']
v21=dt['Brake Pedal Crackoff']
v22=dt['Brake Pedal.1']
v23=dt['Brake Switch.1']
v24=dt['Brake Pedal HS.1']
v25=dt['Clutch Pedal Free Play']
v26=dt['Clutch Pedal Full Stroke']

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20,20))
axes = axes.flatten()

# List of torque measurement columns
torque_cols = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',               'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',               'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',               'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm',               'Brake Pedal_Crackoff_mm', 'Brake Pedal', 'Brake Pedal_Fullstr_mm', 'Brake Switch',               'Brake Pedal HS', 'ClutchPedal FreePlay', 'Clutch Pedal FullStroke', 'Clutch Pedal HS',               'Brake Pedal Crackoff', 'Brake Pedal.1', 'Brake Switch.1', 'Brake Pedal HS.1',               'Clutch Pedal Free Play', 'Clutch Pedal Full Stroke']

colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',]

for i, col in enumerate(torque_cols):
    ax = axes[i]
    ax.scatter(input_var, dt[col], alpha=0.5, color=colors[i%10], label=col)
    ax.set_xlabel('Trace Date')
    ax.set_ylabel('Torque Measurements')
    ax.legend()

plt.tight_layout()
plt.show()

"""Variability In Torque Measurements"""

var1_std = np.std(v1)
var2_std = np.std(v2)
var3_std = np.std(v3)
var4_std = np.std(v4)
var5_std = np.std(v5)
var6_std = np.std(v6)
var7_std = np.std(v7)
var8_std = np.std(v8)
var9_std = np.std(v9)
var10_std = np.std(v10)
var11_std = np.std(v11)
var12_std = np.std(v12)
var13_std = np.std(v13)
var14_std = np.std(v14)
var15_std = np.std(v15)
var16_std = np.std(v16)
var17_std = np.std(v17)
var18_std = np.std(v18)
var19_std = np.std(v19)
var20_std = np.std(v20)
var21_std = np.std(v21)
var22_std = np.std(v22)
var23_std = np.std(v23)
var24_std = np.std(v24)
var25_std = np.std(v25)
var26_std = np.std(v26)

variables = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24', 'v25', 'v26']
std_values = [var1_std, var2_std, var3_std, var4_std, var5_std, var6_std, var7_std, var8_std, var9_std, var10_std, var11_std, var12_std, var13_std, var14_std, var15_std, var16_std, var17_std, var18_std, var19_std, var20_std, var21_std, var22_std, var23_std, var24_std, var25_std, var26_std]

plt.bar(variables, std_values)
plt.xlabel('Torque Measurements')
plt.ylabel('Standard Deviation')
plt.title('Variability in Torque Measurements')
plt.xticks(rotation=90)
plt.show()

"""Based on the visualized graph, it is observed that the variability in torque V26 (Clutch Pedal Full Stroke) is relatively high compared to other torque measurements. This variability can indicate inconsistent or unstable performance of the Clutch Pedal Full Stroke, which may impact the overall functionality and reliability of the system.

Findings:

Inconsistent Torque: The high variability in torque V26 suggests that there may be variations in the force applied during the Clutch Pedal Full Stroke. This could be due to factors such as inconsistent assembly techniques, varying component quality, or improper adjustment of the clutch mechanism.
Potential Quality Issues: The significant variability in torque V26 could be an indication of potential quality issues in the manufacturing process. It is essential to investigate the root causes of this variability to ensure that the clutch system meets the required performance and safety standards.
Performance Impact: The inconsistency in torque V26 can affect the overall performance of the clutch system. It may lead to variations in clutch engagement or disengagement, resulting in an inconsistent driving experience for users. It is crucial to address this issue to ensure smooth operation and optimal performance of the clutch system.

Solutions and Recommendations:

Process Optimization: Review and optimize the assembly process for the Clutch Pedal Full Stroke to ensure consistent and accurate torque application. This may involve providing clear assembly instructions, training operators, and implementing quality control checks at each stage.
Component Evaluation: Evaluate the components related to torque V26, such as clutch mechanism parts, for any defects or variations in quality. Ensure that the components meet the required specifications and standards to minimize variability.
Calibration and Adjustment: Verify and calibrate the clutch system to ensure proper adjustment of the Clutch Pedal Full Stroke. Improper adjustment can contribute to torque variability. Regular maintenance and periodic adjustment may be necessary to maintain consistent performance.
Quality Control Measures: Strengthen quality control measures by implementing more stringent checks and inspections for torque V26. This can include real-time monitoring, statistical process control, and regular audits to identify and address any issues promptly.
Continuous Improvement: Establish a feedback loop for capturing and analyzing torque data over time. Use this data to identify patterns, trends, and areas for improvement. Continuously strive to enhance the design, manufacturing, and assembly processes based on the insights gained from the data analysis.

By addressing the variability in torque V26 and implementing appropriate solutions, it is possible to improve the consistency, performance, and reliability of the Clutch Pedal Full Stroke in the clutch system.
"""

import numpy as np
import matplotlib.pyplot as plt

# Select the relevant columns for pedal play and full stroke measurements
pedal_play = dt[['Pre_Play_mm', 'Brake Pedal_Crackoff_mm', 'ClutchPedal FreePlay', 'Clutch Pedal Free Play']]

# Define the desired ranges for pedal play and full stroke
desired_ranges = {'Pre_Play_mm': (0, 10),
                  'Brake Pedal_Crackoff_mm': (10, 20),
                  'ClutchPedal FreePlay': (0, 5),
                  'Clutch Pedal Free Play': (0, 5)}

# Calculate the deviations from the desired ranges
deviations = pedal_play.apply(lambda x: abs(x - desired_ranges[x.name][1]))

# Set the x-axis labels and bar positions
x_labels = deviations.index
bar_positions = np.arange(len(x_labels))

# Create a stacked bar chart
bar_width = 0.4
fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(len(x_labels))
for i, col in enumerate(deviations.columns):
    ax.bar(bar_positions, deviations[col], bar_width, bottom=bottom, label=col)
    bottom += deviations[col]

ax.set_xlabel('Samples')
ax.set_ylabel('Deviation from Desired Range')
ax.set_title('Pedal Play and Full Stroke Deviations')
ax.set_xticks(bar_positions)
ax.set_xticklabels(x_labels, rotation='vertical')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

"""the measurements related to pedal play and full stroke and visualize the deviations from the desired ranges

the findings from this analysis help in understanding the variability in pedal play and full stroke measurements, identifying problematic areas, and making informed decisions regarding the adjustment and optimization of the pedal system.

bar chart showing the average value of each parameter for the brake and clutch switch states.
"""

import matplotlib.pyplot as plt

# Scatter plot for Brake Switch
fig, ax = plt.subplots()
ax.scatter(dt['Brake Switch'], dt['Brake sw. bracket Torque \n(Screw 1)'], color='blue', label='Brake sw. bracket Torque \n(Screw 1)')
ax.scatter(dt['Brake Switch'], dt['Brake sw. bracket Torque \n(Screw 2)'], color='red', label='Brake sw. bracket Torque \n(Screw 2)')
ax.set_xlabel('Brake Switch')
ax.set_ylabel('Torque Measurements')
ax.legend()
plt.show()

# Bar plot for Clutch Switch
fig, ax = plt.subplots()
ax.bar(dt['Clutch sw. bracket Torque (Screw 1)'], dt['Clutch sw. bracket Torque (Screw 1)'], color='blue', label='Clutch sw. bracket Torque (Screw 1)')
ax.bar(dt['Clutch sw. bracket Torque (Screw 2)'], dt['Clutch sw. bracket Torque (Screw 2)'], color='red', label='Clutch sw. bracket Torque (Screw 2)')
ax.set_xlabel('Clutch Switch')
ax.set_ylabel('Torque Measurements')
ax.legend()
plt.show()

import matplotlib.pyplot as plt

# Filter the relevant columns
switch_columns = ['Brake Switch', 'Clutch sw. bracket Torque (Screw 1)']
other_columns = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                 'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)']

# Create subplots for each parameter
fig, axs = plt.subplots(len(switch_columns), len(other_columns), figsize=(12, 8))

# Iterate over switch columns
for i, switch_column in enumerate(switch_columns):
    # Iterate over other columns
    for j, other_column in enumerate(other_columns):
        # Scatter plot
        axs[i, j].scatter(df[switch_column], df[other_column], alpha=0.5)
        axs[i, j].set_xlabel(switch_column)
        axs[i, j].set_ylabel(other_column)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# Get the data for brake and clutch switches
brake_switch = dt['Brake Switch']
clutch_switch = dt['Clutch sw. bracket Torque (Screw 1)']

# Define the parameters to compare with brake and clutch switches
parameters = ['BPT1', 'BPT2', 'CPT1', 'CPT2', 'DT1', 'DT2', 'DT3', 'DT4', 'CT1', 'CT2', 'PV', 'TR1BPCR', 'TR1BPFT', 'TR1CPFP', 'TR1CPFS', 'TR2BPCR', 'TR2BPFT', 'TR2CPFP', 'TR2CPFS']

# Get the average values for each parameter based on the brake and clutch switch states
brake_switch_avg = [dt.loc[brake_switch == 1, param].mean() for param in parameters]
clutch_switch_avg = [dt.loc[clutch_switch == 1, param].mean() for param in parameters]

# Set up the bar chart
bar_width = 0.35
index = np.arange(len(parameters))

# Plot the grouped bars
fig, ax = plt.subplots(figsize=(12, 6))
brake_bars = ax.bar(index, brake_switch_avg, bar_width, label='Brake Switch')
clutch_bars = ax.bar(index + bar_width, clutch_switch_avg, bar_width, label='Clutch Switch')

# Set labels and titles
ax.set_xlabel('Parameters')
ax.set_ylabel('Average Value')
ax.set_title('Average Value of Parameters based on Brake and Clutch Switches')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(parameters, rotation=45, ha='right')
ax.legend()

# Display the chart
plt.tight_layout()
plt.show()

"""Based on the provided finding that on 2022-11-07, the brake switch measurement is at the top with a value of 110, here are some possible inferences:

Anomalies in Brake Switch Measurement: The high value of 110 for the brake switch measurement on 2022-11-07 could indicate a potential anomaly or abnormality in the functioning of the brake switch on that particular day. Further investigation might be required to determine the cause of this unusually high measurement.
Potential Brake System Issue: The elevated brake switch measurement suggests that there might be an issue with the brake system on the given date. It could indicate a malfunctioning brake switch, a problem with the brake pedal or its associated components, or an issue with the brake fluid or hydraulic system.
Maintenance or Inspection Required: The finding highlights the importance of regular maintenance and inspections of the brake system. It indicates the need to thoroughly check and potentially replace the brake switch or other relevant components to ensure the proper functioning and reliability of the brake system.
Data Validation: The high brake switch measurement could also raise questions about the accuracy and reliability of the data collected on that specific date. It is important to verify the measurement and ensure that it is not due to measurement errors or data recording issues.
Further Analysis: This finding should prompt a deeper analysis of the data to identify any correlations or patterns between the brake switch measurement and other relevant parameters. It can help uncover potential relationships or dependencies that might provide additional insights into the brake system's performance and reliability.
It is important to note that these inferences are speculative based on the provided information. Further analysis and domain-specific knowledge are required to validate and draw more accurate conclusions from the findings.
"""

import matplotlib.pyplot as plt

# Convert the 'Scan Date' column to datetime format
dt['Scan Date'] = pd.to_datetime(dt['Scan Date'])

# Create a combined column for scan date and time
dt['Scan DateTime'] = dt['Scan Date'] + pd.to_timedelta(dt['Scan Time'])

# Sort the DataFrame by scan date and time
dt = dt.sort_values('Scan DateTime')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(dt['Scan DateTime'], dt['Brake Switch'], marker='o', linestyle='-', markersize=4)

# Customize the plot
plt.xlabel('Scan Date and Time')
plt.ylabel('Brake Switch')
plt.title('Distribution of Brake Switch over Scan Date and Time')
plt.xticks(rotation=45)
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Convert the 'Scan Date' column to datetime format
dt['Scan Date'] = pd.to_datetime(dt['Scan Date'])

# Create a combined column for scan date and time
dt['Scan DateTime'] = dt['Scan Date'] + pd.to_timedelta(dt['Scan Time'])

# Sort the DataFrame by scan date and time
dt = dt.sort_values('Scan DateTime')

# Find the maximum brake switch measurement on November 7, 2022
max_measurement = dt.loc[dt['Scan Date'] == '2022-11-07', 'Brake Switch'].max()
max_measurement_index = dt.loc[(dt['Scan Date'] == '2022-11-07') & (dt['Brake Switch'] == max_measurement)].index[0]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(dt['Scan DateTime'], dt['Brake Switch'], marker='o', linestyle='-', markersize=4)

# Annotate the point with the highest brake switch measurement
plt.annotate(f'Max Measurement: {max_measurement}', xy=(dt['Scan DateTime'][max_measurement_index], max_measurement),
             xytext=(10, 30), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

# Customize the plot
plt.xlabel('Scan Date and Time')
plt.ylabel('Brake Switch')
plt.title('Distribution of Brake Switch over Scan Date and Time')
plt.xticks(rotation=45)
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Extract the relevant columns from the dataframe
materials = dt['Material']
brake_torque_screw1 = dt['Brake sw. bracket Torque \n(Screw 1)']

# Get the unique materials and their corresponding counts
unique_materials, material_counts = np.unique(materials, return_counts=True)

# Calculate the mean brake torque for each material
mean_brake_torque = []
for material in unique_materials:
    mean_brake_torque.append(np.mean(brake_torque_screw1[materials == material]))

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(unique_materials, mean_brake_torque)
plt.xlabel('Material')
plt.ylabel('Mean Brake Torque (Screw 1)')
plt.title('Mean Brake Torque (Screw 1) Across Different Materials')
plt.xticks(rotation=45)
plt.show()





"""Variability in Torque Measurements:"""

import matplotlib.pyplot as plt

# Define the torque measurement columns
torque_columns = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                  'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                  'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                  'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)']

# Calculate the variability (standard deviation) for each torque measurement
variability = dt[torque_columns].std()

# Plot the variability
plt.figure(figsize=(10, 6))
variability.plot(kind='bar', color='blue')
plt.xlabel('Torque Measurements')
plt.ylabel('Variability')
plt.title('Variability in Torque Measurements')
plt.xticks(rotation=45)
plt.show()

"""This high variability in "DBV torque 4" suggests that the values for this measurement may vary significantly from one observation to another. It could be an indication of inconsistency or instability in the torque applied to the "DBV torque 4" component. This variability could arise from various factors such as differences in the tightening process, variations in the components used, or other external factors affecting the torque application.

Identifying the torque measurement with the highest variability is valuable in quality control and process optimization. It highlights the need for further investigation and potential improvement actions to reduce the variability and ensure consistent torque application. By addressing the factors contributing to the high variability in "DBV torque 4," steps can be taken to enhance the torque control process, improve the reliability of the component, and minimize potential issues or failures associated with inconsistent torque

Pedal Play and Full Stroke:
"""

import matplotlib.pyplot as plt

# Define the pedal play and full stroke columns
pedal_columns = ['Pre_Play_mm', 'Brake Pedal_Crackoff_mm', 'ClutchPedal FreePlay', 'Clutch Pedal Free Play']

# Plot the pedal play and full stroke measurements
plt.figure(figsize=(10, 6))
dt[pedal_columns].boxplot()
plt.xlabel('Pedal Measurements')
plt.ylabel('Measurement (mm)')
plt.title('Pedal Play and Full Stroke Measurements')
plt.xticks(rotation=45)
plt.show()

"""it appears that the "Brake Pedal_Crackoff_mm" measurement has a relatively high value compared to the other pedal measurements. This can be observed from the position of the upper whisker of the box plot, indicating a higher maximum value for this measurement.

Additionally, the "Clutch Pedal Free Play" measurement has some outliers, as indicated by the individual points outside the whiskers. Outliers represent data points that deviate significantly from the majority of the dataset. In this case, the presence of outliers suggests that there are some extreme values or variations in the "Clutch Pedal Free Play" measurement that are different from the rest of the dataset.

These observations are valuable for identifying potential issues or variations in the pedal measurements. Further investigation and analysis may be required to understand the reasons behind the higher measurement in "Brake Pedal_Crackoff_mm" and the presence of outliers in "Clutch Pedal Free Play". This information can help in addressing any problems or optimizing the pedal performance and quality control processes.

Brake and Clutch Switches:
"""

import matplotlib.pyplot as plt

# Define the brake and clutch switch columns
switch_columns = ['Brake Switch', 'Clutch sw. bracket Torque (Screw 1)']

# Count the occurrences of each switch state
switch_counts = dt[switch_columns].apply(pd.value_counts)

# Plot the switch states
plt.figure(figsize=(12, 8))  # Increase the size of the chart (width: 12, height: 8)
switch_counts.plot(kind='bar', stacked=True)
plt.xlabel('Switch States')
plt.ylabel('Count')
plt.title('Brake and Clutch Switch States')
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small'
)
max_count = switch_counts.values[~np.isnan(switch_counts.values)].max()
plt.ylim(0, max_count + 1)  # Set the y-axis limits based on the maximum count value
plt.tight_layout()  # Adjust the layout for better spacing
plt.show()

"""Scan Date and Time:"""

import matplotlib.pyplot as plt

# Extract the scan date and time columns
scan_date = pd.to_datetime(dt['Scan Date'])
scan_time = pd.to_datetime(dt['Scan Time'])

# Plot the distribution of scan dates
plt.figure(figsize=(10, 6))
plt.hist(scan_date, bins='auto')
plt.xlabel('Scan Date')
plt.ylabel('Frequency')
plt.title('Distribution of Scan Dates')
plt.xticks(rotation=45)
plt.show()

# Plot the distribution of scan times
plt.figure(figsize=(10, 6))
plt.hist(scan_time, bins='auto')
plt.xlabel('Scan Time')
plt.ylabel('Frequency')
plt.title('Distribution of Scan Times')
plt.xticks(rotation=45)
plt.show()

"""it appears that the date "2022-11-17" has a higher frequency compared to other dates in the dataset. This can be observed in the first histogram, which shows the distribution of scan dates.

The higher frequency of the date "2022-11-17" suggests that there might be a specific reason or event associated with that particular date. It could indicate a concentrated period of scanning activity or a significant event that led to more scans being performed on that day.

it appears that the scan time "05-02 16" has a higher frequency compared to other scan times in the dataset. This can be observed in the second histogram, which shows the distribution of scan times.

The higher frequency of the scan time "05-02 16" suggests that there might be a specific period during the day when more scans were conducted. It could indicate a particular shift, work schedule, or operational pattern that resulted in an increased number of scans at that time.

Material and Batch Information:
"""

import matplotlib.pyplot as plt

# Extract the material and batch columns
materials = dt['Material']
batches = dt['Batch']

# Count the occurrences of each material
material_counts = materials.value_counts()

# Count the occurrences of each batch
batch_counts = batches.value_counts()

# Plot the material counts
plt.figure(figsize=(10, 6))
material_counts.plot(kind='bar', color='blue')
plt.xlabel('Material')
plt.ylabel('Count')
plt.title('Material')

"""analysis of the scan date and time in process traceability"""

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your dataset is stored in a DataFrame called 'dt'
# Replace 'column_name' with the actual column name for scan dates
scan_dates = pd.to_datetime(dt['Scan Date'])

# Analyzing Temporal Trends
scan_dates.value_counts().sort_index().plot(kind='line')
plt.xlabel('Scan Date')
plt.ylabel('Number of Scans')
plt.title('Temporal Trends of Scans')
plt.show()

# Analyzing Anomalies or Outliers
scan_times = pd.to_datetime(dt['Scan Time'])
scan_times.hist()
plt.xlabel('Scan Time')
plt.ylabel('Frequency')
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small'
)
plt.title('Distribution of Scan Times')
plt.show()

# Data Consistency
scan_dates.groupby(scan_dates.dt.date).nunique().plot(kind='line')
plt.xlabel('Scan Date')
plt.ylabel('Number of Unique Scan Times')
plt.xticks(
    rotation=45,
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small'
)
plt.title('Data Consistency over Time')
plt.show()

# Process Efficiency
scan_diff = scan_dates.diff().astype('timedelta64[m]')
scan_diff.plot(kind='line')
plt.xlabel('Scan Date')
plt.ylabel('Time Gap (minutes)')
plt.title('Time Gaps between Scans')
plt.show()

import pandas as pd

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")

# Torque Analysis
torque_columns = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                  'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                  'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                  'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)']
torque_variability = df[torque_columns].std()
print("Torque Variability:")
print(torque_variability)

# Pedal Stroke Analysis
pedal_columns = ['Pre_Play_mm', 'Brake Pedal_Crackoff_mm', 'Brake Pedal_Fullstr_mm', 'ClutchPedal FreePlay']
pedal_stroke = df[pedal_columns]
print("Pedal Stroke Analysis:")
print(pedal_stroke.describe())

# Switch Analysis
switch_columns = ['Brake Switch']
switch_states = df[switch_columns]
print("Switch Analysis:")
print(switch_states.value_counts())

# Assembly Process Traceability
trace_columns = ['Material', 'Shift Indicator', 'Trace Serial No', 'Trace Date', 'Batch', 'Scan Date', 'Scan Time']
assembly_traceability = df[trace_columns]
print("Assembly Process Traceability:")
print(assembly_traceability)

# Additional Analysis and Inferences
# You can add further analysis and inference code based on your specific requirements

"""Findings:

Torque Variability: The torque measurements for different components show variations, indicating inconsistencies or variations in the manufacturing or assembly process. This may lead to quality issues and potential performance problems in the vehicle control systems.

Pedal Stroke Analysis: The analysis reveals variations in pre-play, crack-off, and full stroke measurements of the pedals. Inconsistencies in these measurements can affect pedal functionality and driver comfort.

Switch Analysis: The frequency distribution of brake switch states indicates potential variations or issues with the brake system's operation and reliability.

Assembly Process Traceability: The data provides detailed information about the assembly process, including material used, shift indicators, trace serial numbers, and batch numbers. This traceability information enables tracking and monitoring of the assembly process.

Problems:

Torque Variability: Inconsistent torque measurements can lead to variations in the performance and reliability of vehicle components. This can result in quality issues and potential safety concerns.

Pedal Stroke Variations: Inconsistent pedal stroke measurements can affect the overall performance and driver experience. Inaccurate pedal play and stroke can impact braking and clutch engagement, compromising vehicle control.

Switch Reliability: The presence of variations in brake switch states suggests potential issues with switch functionality or reliability, which can affect the brake system's operation and safety.

Solutions:

Torque Variability: Identify the root causes of torque variations and take corrective actions in the manufacturing and assembly processes. This may involve improving assembly techniques, tightening procedures, or identifying faulty components.

Pedal Stroke Variations: Establish clear specifications for pre-play, crack-off, and full stroke measurements. Implement consistent assembly and adjustment procedures to ensure accurate and reliable pedal functionality.

Switch Reliability: Conduct thorough inspections and testing of brake switches to identify and replace any faulty or unreliable switches. Implement regular maintenance and quality checks to ensure switch reliability and proper brake system operation.

Assembly Process Traceability: Utilize the traceability information to monitor and track the assembly process. This enables identification of any inconsistencies or issues in specific batches or manufacturing shifts, facilitating timely corrective actions and quality improvements.

Implementing these solutions will help address the identified findings, improve consistency and reliability in the assembly process, and enhance the overall quality and performance of the commercial vehicle solutions.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")

# Torque Analysis
torque_columns = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                  'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                  'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                  'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)']
torque_variability = df[torque_columns].std()
print("Torque Variability:")
print(torque_variability)

# Pedal Stroke Analysis
pedal_columns = ['Pre_Play_mm', 'Brake Pedal_Crackoff_mm', 'Brake Pedal_Fullstr_mm', 'ClutchPedal FreePlay']
pedal_stroke = df[pedal_columns]
print("Pedal Stroke Analysis:")
print(pedal_stroke.describe())

# Visualize Pedal Stroke Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(data=pedal_stroke)
plt.title("Pedal Stroke Analysis")
plt.xlabel("Measurements")
plt.ylabel("Value (mm)")
plt.show()

# Switch Analysis
switch_columns = ['Brake Switch']
switch_states = df[switch_columns]
print("Switch Analysis:")
print(switch_states.value_counts())

# Visualize Switch Analysis
plt.figure(figsize=(10, 6))
sns.countplot(data=switch_states, x='Brake Switch')
plt.title("Switch Analysis - Brake Switch")
plt.xlabel("Switch State")
plt.ylabel("Count")
plt.show()

# Assembly Process Traceability
trace_columns = ['Material', 'Shift Indicator', 'Trace Serial No', 'Trace Date', 'Batch', 'Scan Date', 'Scan Time']
assembly_traceability = df[trace_columns]
print("Assembly Process Traceability:")
print(assembly_traceability)

# Additional Analysis and Inferences
# You can add further analysis and inference code based on your specific requirements

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")

# Comparing Torque Values
torque_columns = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                  'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                  'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                  'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)']
torque_comparison = df[torque_columns].mean()
print("Comparing Torque Values:")
print(torque_comparison)

# Outliers
fig, axs = plt.subplots(ncols=len(torque_columns), figsize=(20, 5))
for i, column in enumerate(torque_columns):
    sns.boxplot(x=df[column], ax=axs[i])
    axs[i].set_title(column)
plt.show()

# Quality Control
torque_variability = df[torque_columns].std()
print("Torque Variability over time:")
print(torque_variability)

# Compliance with Specifications
torque_specifications = {'Brake sw. bracket Torque \n(Screw 1)': [10, 20],
                         'Brake sw. bracket Torque \n(Screw 2)': [10, 20],
                         'Clutch sw. bracket Torque (Screw 1)': [15, 25],
                         'Clutch sw. bracket Torque (Screw 2)': [15, 25],
                         'DBV torque (Bolt 1)': [25, 35],
                         'DBV torque (Bolt 2)': [25, 35],
                         'DBV torque (Bolt 3)': [25, 35],
                         'DBV torque (Bolt 4)': [25, 35],
                         'CMC Torque (Bolt 1)': [30, 40],
                         'CMC Torque (Bolt 2)': [30, 40]}
for column, specification in torque_specifications.items():
    min_spec = specification[0]
    max_spec = specification[1]
    out_of_spec = df[(df[column] < min_spec) | (df[column] > max_spec)]
    print(f"{column} outside of specification:")
    print(out_of_spec)

import pandas as pd

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")

# Select the input and output parameters
input_parameters = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                    'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                    'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                    'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_parameters = ['Brake Pedal_Crackoff_mm', 'Brake Pedal', 'Brake Pedal_Fullstr_mm', 'Brake Switch',
                     'Brake Pedal HS', 'ClutchPedal FreePlay', 'Clutch Pedal FullStroke', 'Clutch Pedal HS',
                     'Brake Pedal Crackoff', 'Brake Pedal.1', 'Brake Pedal Full Stroke', 'Brake Switch.1',
                     'Brake Pedal HS.1', 'Clutch Pedal Free Play', 'Clutch Pedal Full Stroke']

# Calculate the correlation between input and output parameters
correlation_matrix = df[input_parameters + output_parameters].corr()

# Analyze the correlation matrix
input_output_corr = correlation_matrix.loc[input_parameters, output_parameters]

# Find the output parameter with the highest correlation for each input parameter
best_output_parameter = input_output_corr.idxmax()

print("Correlation between input and output parameters:")
print(input_output_corr)

print("Best output parameter for each input parameter:")
print(best_output_parameter)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")

# Select the input and output parameters
input_parameters = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                    'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                    'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                    'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_parameters = ['Brake Pedal_Crackoff_mm', 'Brake Pedal', 'Brake Pedal_Fullstr_mm', 'Brake Switch',
                     'Brake Pedal HS', 'ClutchPedal FreePlay', 'Clutch Pedal FullStroke', 'Clutch Pedal HS',
                     'Brake Pedal Crackoff', 'Brake Pedal.1', 'Brake Pedal Full Stroke', 'Brake Switch.1',
                     'Brake Pedal HS.1', 'Clutch Pedal Free Play', 'Clutch Pedal Full Stroke']

# Calculate the correlation between input and output parameters
correlation_matrix = df[input_parameters + output_parameters].corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation between Input and Output Parameters")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")

# Select the input and output parameters
input_parameters = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                    'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                    'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                    'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_parameters = ['Brake Pedal_Crackoff_mm', 'Brake Pedal', 'Brake Pedal_Fullstr_mm', 'Brake Switch',
                     'Brake Pedal HS', 'ClutchPedal FreePlay', 'Clutch Pedal FullStroke', 'Clutch Pedal HS',
                     'Brake Pedal Crackoff', 'Brake Pedal.1', 'Brake Pedal Full Stroke', 'Brake Switch.1',
                     'Brake Pedal HS.1', 'Clutch Pedal Free Play', 'Clutch Pedal Full Stroke']

# Create scatter plots to visualize the relationship between input and output parameters
plt.figure(figsize=(10, 6))
for input_param in input_parameters:
    plt.scatter(df[input_param], df[output_parameters[0]], label=input_param)

plt.xlabel("Input Parameters")
plt.ylabel(output_parameters[0])
plt.legend()
plt.title(f"Scatter plot of {output_parameters[0]} against Input Parameters")
plt.show()

import pandas as pd

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")

# Select the input and output parameters
input_parameters = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                    'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                    'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                    'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_parameters = ['Brake Pedal_Crackoff_mm', 'Brake Pedal', 'Brake Pedal_Fullstr_mm', 'Brake Switch',
                     'Brake Pedal HS', 'ClutchPedal FreePlay', 'Clutch Pedal FullStroke', 'Clutch Pedal HS',
                     'Brake Pedal Crackoff', 'Brake Pedal.1', 'Brake Pedal Full Stroke', 'Brake Switch.1',
                     'Brake Pedal HS.1', 'Clutch Pedal Free Play', 'Clutch Pedal Full Stroke']

# Calculate the correlation between input and output parameters
correlation_matrix = df[input_parameters + output_parameters].corr()

# Find the output parameter with the highest correlation for each input parameter
best_output_parameter = correlation_matrix.loc[input_parameters].idxmax()

print("Best output parameter for each input parameter:")
print(best_output_parameter)

import pandas as pd
import matplotlib.pyplot as plt

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")

# Select the input and output parameters
input_parameters = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                    'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                    'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                    'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_parameters = ['Brake Pedal_Crackoff_mm', 'Brake Pedal', 'Brake Pedal_Fullstr_mm', 'Brake Switch',
                     'Brake Pedal HS', 'ClutchPedal FreePlay', 'Clutch Pedal FullStroke', 'Clutch Pedal HS',
                     'Brake Pedal Crackoff', 'Brake Pedal.1', 'Brake Pedal Full Stroke', 'Brake Switch.1',
                     'Brake Pedal HS.1', 'Clutch Pedal Free Play', 'Clutch Pedal Full Stroke']

# Calculate the correlation between input and output parameters
correlation_matrix = df[input_parameters + output_parameters].corr()

# Find the best output parameter for each input parameter
best_output_parameter = correlation_matrix.loc[input_parameters, output_parameters].idxmax(axis=1)

# Generate the bar plot
plt.figure(figsize=(10, 6))
plt.bar(input_parameters, best_output_parameter)
plt.xlabel("Input Parameters")
plt.ylabel("Best Output Parameter")
plt.title("Best Output Parameter for Each Input Parameter")
plt.xticks(rotation=90)
plt.show()

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")
df.dropna(inplace=True)  # Drop rows with missing values


# Select the input and output parameters
input_parameters = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
                    'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
                    'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
                    'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_parameters = ['Brake Pedal_Crackoff_mm', 'Brake Pedal', 'Brake Pedal_Fullstr_mm', 'Brake Switch',
                     'Brake Pedal HS', 'ClutchPedal FreePlay', 'Clutch Pedal FullStroke', 'Clutch Pedal HS',
                     'Brake Pedal Crackoff', 'Brake Pedal.1', 'Brake Pedal Full Stroke', 'Brake Switch.1',
                     'Brake Pedal HS.1', 'Clutch Pedal Free Play', 'Clutch Pedal Full Stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[input_parameters], df[output_parameters], test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
score = model.score(X_test, y_test)
print("Model Score:", score)

"""A negative R^2 score suggests that the linear regression model is not able to effectively capture the relationship between the input variables and the target variable. An R^2 score of -0.05097861834303085 indicates that the model is performing poorly and is not able to explain the variance in the target variable based on the input variables.

There are several possible reasons for this poor performance:

Non-linear relationship: Linear regression assumes a linear relationship between the input variables and the target variable. If the relationship is non-linear, a linear regression model may not be appropriate. You could try using other regression models such as polynomial regression or decision tree regression that can capture non-linear relationships.
Insufficient features: The input variables you have selected may not be sufficient to accurately predict the target variable. You could consider adding more relevant features or performing feature engineering to create additional meaningful features.
Multicollinearity: If the selected input variables are highly correlated with each other, it can lead to multicollinearity issues, which can negatively impact the performance of the linear regression model. In such cases, you could consider removing highly correlated variables or using regularization techniques such as Ridge regression or Lasso regression.
Outliers or missing data: Outliers or missing data in the dataset can affect the performance of the model. It's important to handle outliers appropriately and address any missing data through imputation or removal.
Insufficient data: The performance of the model can be affected by the size of the dataset. If you have a small dataset, it may not provide enough information for the model to learn the underlying patterns. Gathering more data or using techniques such as cross-validation can help mitigate this issue.
It's recommended to explore these possibilities and evaluate different models and techniques to improve the performance of the prediction. Additionally, consider evaluating the model using other metrics such as mean squared error (MSE) or mean absolute error (MAE) to get a better understanding of the model's performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")
df.dropna(inplace=True)  # Drop rows with missing values


# Select input and output variables
input_cols = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
              'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
              'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
              'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_cols = ['Brake Pedal_Crackoff_mm', 'Brake Pedal', 'Brake Pedal_Fullstr_mm', 'Brake Switch',
               'Brake Pedal HS', 'ClutchPedal FreePlay', 'Clutch Pedal FullStroke', 'Clutch Pedal HS',
               'Brake Pedal Crackoff', 'Brake Pedal.1', 'Brake Pedal Full Stroke', 'Brake Switch.1',
               'Brake Pedal HS.1', 'Clutch Pedal Free Play', 'Clutch Pedal Full Stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[input_cols], df[output_cols], test_size=0.2, random_state=42)

# Create a pipeline with feature engineering and model
pipeline = Pipeline([
    ('polynomial_features', PolynomialFeatures(degree=2)),
    ('ridge_regression', Ridge(alpha=1, solver="cholesky"))
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Perform cross-validation
cv_scores = cross_val_score(pipeline, df[input_cols], df[output_cols], cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()
print("Cross-Validation Mean Squared Error:", cv_mse)

"""The mean squared error (MSE) on the test set is 789.45, which measures the average squared difference between the predicted values and the actual values. The lower the MSE, the better the model's performance.

The cross-validation mean squared error (CV MSE) is 833.54, which is the average MSE obtained through cross-validation. Cross-validation helps estimate how well the model is likely to perform on unseen data. In this case, the CV MSE is slightly higher than the MSE on the test set, indicating that the model's performance may vary slightly when applied to different subsets of the data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")
df.dropna(inplace=True)  # Drop rows with missing values


# Select input and output variables
input_cols = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
              'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
              'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
              'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_col = 'Brake Pedal_Crackoff_mm'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[input_cols], df[output_col], test_size=0.2, random_state=42)

# Apply appropriate preprocessing techniques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = y_train.ravel()
y_test = y_test.ravel()


# Regularization
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge = Ridge()
grid = GridSearchCV(ridge, {'alpha': alphas}, cv=5)
grid.fit(X_train_scaled, y_train)
best_alpha = grid.best_params_['alpha']

# Train a Ridge regression model with the optimal alpha
ridge = Ridge(alpha=best_alpha)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

# Feature Selection
selector = SelectFromModel(estimator=ridge, threshold='mean')
selector.fit(X_train_scaled, y_train)
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Model Selection
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_selected, y_train)
y_pred = rf.predict(X_test_selected)
mse_rf = mean_squared_error(y_test, y_pred)

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train_selected, y_train)
y_pred = gb.predict(X_test_selected)
mse_gb = mean_squared_error(y_test, y_pred)

if mse_rf < mse_gb:
    best_model = rf
    best_mse = mse_rf
else:
    best_model = gb
    best_mse = mse_gb

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6],
    'min_samples_split': [2, 4, 8]
}
grid = GridSearchCV(best_model, param_grid, cv=5)
grid.fit(X_train_selected, y_train)
best_params = grid.best_params_

# Train the final model with the best hyperparameters
final_model = best_model.set_params(**best_params)
final_model.fit(X_train_selected, y_train)
y_pred = final_model.predict(X_test_selected)
final_mse = mean_squared_error(y_test, y_pred)
print(final_mse)

"""feature engineering using regularization, feature selection using the Ridge regression model, model selection between Random Forest and Gradient Boosting, and hyperparameter tuning using GridSearchCV. It also incorporates data preprocessing using StandardScaler to handle scaling of the input features."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Read the data into a DataFrame
df = pd.read_csv("/Users/santhoshkumar/Downloads/PTSetconnect/racprod.csv")
df.dropna(inplace=True)  # Drop rows with missing values


# Select input and output variables
input_cols = ['Brake sw. bracket Torque \n(Screw 1)', 'Brake sw. bracket Torque \n(Screw 2)',
              'Clutch sw. bracket Torque (Screw 1)', 'Clutch sw. bracket Torque (Screw 2)',
              'DBV torque (Bolt 1)', 'DBV torque (Bolt 2)', 'DBV torque (Bolt 3)', 'DBV torque (Bolt 4)',
              'CMC Torque (Bolt 1)', 'CMC Torque (Bolt 2)', 'Pedal shaft_Assy_Torq', 'Pre_Play_mm']

output_col = 'Brake Pedal_Crackoff_mm'  # Choose one output column as the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[input_cols], df[output_col], test_size=0.2, random_state=42)

# Reshape y_train and y_test to 1-dimensional arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Train the Ridge regression model
ridge = Ridge()
ridge.fit(X_train, y_train)

# Predict on the test set
y_pred = ridge.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

"""The mean squared error (MSE) of 30.53 indicates the average squared difference between the predicted and actual values of the target variable. A lower MSE suggests that the model's predictions are closer to the actual values, indicating better performance.


"""
