import os
import pandas as pd
import glob
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy import stats
# use your path
path = '/your directory for CNTL animals csv files/'
all_files = glob.glob(os.path.join(path, "*.csv"))
df_single = []
for f in all_files:
    df_primary = pd.read_csv(f)
    # the following multiplied number
    df = df_primary.drop(df_primary.index[np.where(df_primary.index > 9000)])
    df['Dis_cm_sum'] = df['Distance_cm'].cumsum()
    df2 = df[['Frame', 'ROI_location', 'ROI_transition', 'Distance_cm', 'Dis_cm_sum']]
    df_single.append(df2)
    # Concatenate the dataframes 
    s = pd.concat(df_single, axis=0)

# set up the figure
sns.set_theme(style="ticks")
plt.rcParams.update({'font.size': 7})

# make the global figure size to 10 inches wide and 6 inches tall
plt.figure(figsize=(10, 6))

# select the rows where the 'ROI_location' column is 
selected_rows1 = s['ROI_location'].isin(['c1','c2','c3','c4'])
s_selected1 = s[selected_rows1]

# pick the "Dis_m_Sum" and "Velocity" columns
x1 = s_selected1["Frame"]
y1 = s_selected1["Dis_cm_sum"]
# for the Pearson correlation coefficient (r) and p-value for the first dataset
r, p = pearsonr(x1, y1)
# set up the figure
f, ax = plt.subplots(figsize=(8, 8))
# pick the rows where the 'ROI_location' column is 
selected_rows2 = s['ROI_location'].isin(['center'])
s_selected2 = s[selected_rows2]

# choose the "Dis_m_Sum" and "Velocity" columns for the second dataset
x2 = s_selected2["Frame"]
y2 = s_selected2["Dis_cm_sum"]
# Calculate the Pearson correlation coefficient (r) and p-value for the second dataset
r2, p2 = pearsonr(x2, y2)

# make the first scatter plot
sns.scatterplot(x=x1, y=y1, color=(1, 0, 0, 0.1), s=.001, edgecolor="none", alpha=0.0001, ax=ax, )

# plot the second scatter plot
sns.scatterplot(x=x2, y=y2, color=(0, 0, 1, 0.5), s=.001, edgecolor="none", alpha=0.0001, ax=ax)
# Set the x-axis and y-axis labels
ax.set_xlabel("Frame (fps)")
ax.set_ylabel("Dis_cm_sum")

# initialize the x and y axis limits for drawng better plots
x_min = 0
x_max = 0
y_min = 0
y_max = 0

# iterate over the x and y variables
i = 1
while True:
    try:
        x = eval(f"x{i}")
        y = eval(f"y{i}")
        # Update the axis limits to include the current x and y data
        x_min = int(min(x_min, min(x)))
        x_max = int(max(x_max, max(x)))
        y_min = int(min(y_min, min(y)))
        y_max = int(max(y_max, max(y)))
        i += 1
    except NameError:
        break
# fit a linear regression model to the first dataset
slope, intercept, r_value, p_value, std_err = stats.linregress(x1, y1)
# calculate the y-values for the fitted line
y_pred = slope * x1 + intercept
# plot the first scatter plot and linear regression
sns.regplot(x=x1, y=y1, color="red", ax=ax, label="Corner", line_kws={"linewidth": 1.0})

#  the equation of the fitted line to the plot
plt.annotate(f"Y1 = {slope:.4f}X1 + {intercept:.2f}", xy=(1.01, 0.88), xycoords="axes fraction", size=11, ha="left", color="red")
plt.text(1.1, 0.85, f"p-value: {p_value:.5f}", transform=plt.gca().transAxes, ha="left", color="red")
plt.text(1.1, 0.82, f"standard error: {std_err:.3f}", transform=plt.gca().transAxes, ha="left", color="red")
# r value to the plot as text
ax.text(1.01, 0.92, f"R1 = {r:.3f}", transform=ax.transAxes, fontsize=11)
# fit a linear regression model to the second dataset
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)

# consider the y-values for the fitted line
y_pred2 = slope2 * x2 + intercept2

# plot the second scatter plot and linear regression
# set the edge color of the line
sns.regplot(x=x2, y=y2, color="blue", ax=ax, label="Center", line_kws={"color": "blue", "linewidth": 1.3, "zorder": 15})

#  the equation of the fitted line to the plot
plt.annotate(f"Y2 = {slope2:.4f}X2 + {intercept2:.2f}", xy=(1.01, 0.7), xycoords="axes fraction", size=11, ha="left", color="blue")
#  the p-value and standard error to the plot
plt.text(1.1, 0.67, f"p-value: {p_value2:.5f}", transform=plt.gca().transAxes, ha="left", color="blue")
plt.text(1.1, 0.64, f"standard error: {std_err2:.3f}", transform=plt.gca().transAxes, ha="left", color="blue")
# the r value to the plot as text
ax.text(1.01, 0.73, f"R2 = {r2:.3f}", transform=ax.transAxes, fontsize=11)
# Create the plot
sns.lineplot(x=[x_min, x_max], y=[y_min, y_max], color="gray", linestyle="dashed", linewidth=.99, label="Bisector of 90"+'\u00b0')

# set the x and y axis limits to include the plot frames
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# Add a legend to the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# set the tick labels for the x and y axes
ax.set_yticklabels(range(y_min, y_max+1))
ax.set_xticklabels(range(x_min, x_max+1))
plt.show()
#for  drug section set the directory and add files of druged animals
path = '/Users/your second file contains csv files of druged animals/'
all_files = glob.glob(os.path.join(path, "*.csv"))
df_single_drug = []
for f in all_files:
    df_primary_drug = pd.read_csv(f)
    # the following multiplied number 3/10 is representing the coefficient of 1 second equal with 30 frame and 1cm=1/100 meter
    dfdrug = df_primary_drug.drop(df_primary_drug.index[np.where(df_primary_drug.index > 9000)])
    dfdrug['Dis_cm_sum'] = dfdrug['Distance_cm'].cumsum()
    df3drug = dfdrug[['Frame', 'ROI_location', 'ROI_transition', 'Distance_cm', 'Dis_cm_sum']]
    df_single_drug.append(df3drug)
    # Concatenate the dataframes and save the result to a CSV file
    s_drug = pd.concat(df_single_drug, axis=0)
# choose the rows where the 'ROI_location' column is 
selected_rows1_drug = s_drug['ROI_location'].isin(['c1','c2','c3','c4'])
s_drug_selected1 = s_drug[selected_rows1_drug]

# select the "Dis_m_Sum" and "Velocity" columns
x3 = s_drug_selected1["Frame"]
y3 = s_drug_selected1["Dis_cm_sum"]
# calculate the Pearson correlation coefficient (r) and p-value for the first dataset
r, p = pearsonr(x3, y3)
# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
# select the rows where the 'ROI_location' column is 
selected_rows2_drug = s_drug['ROI_location'].isin(['center'])
s_drug_selected2 = s_drug[selected_rows2_drug]

# select the "Dis_m_Sum" and "Velocity" columns for the second dataset
x4 = s_drug_selected2["Frame"]
y4 = s_drug_selected2["Dis_cm_sum"]
# calculate the Pearson correlation coefficient (r) and p-value for the second dataset
r2, p2 = pearsonr(x4, y4)

# plot the first scatter plot
sns.scatterplot(x=x3, y=y3, color=(1, 0, 0, 0.1), s=.001, edgecolor="none", alpha=0.0001, ax=ax, )

# plot the second scatter plot
sns.scatterplot(x=x4, y=y4, color=(0, 0, 1, 0.5), s=.001, edgecolor="none", alpha=0.0001, ax=ax)
# Set the x-axis and y-axis labels
ax.set_xlabel("Frame (fps)")
ax.set_ylabel("Dis_cm_sum")

# fit a linear regression model to the first dataset
slope, intercept, r_value, p_value, std_err = stats.linregress(x3, y3)

# calculate the y-values for the fitted line
y_pred = slope * x3 + intercept

# plot the first scatter plot and linear regression
sns.regplot(x=x3, y=y3, color="red", ax=ax, label="Corner", line_kws={"linewidth": 1.0})

# add the equation of the fitted line to the plot
plt.annotate(f"Y1 = {slope:.4f}X1 + {intercept:.2f}", xy=(1.01, 0.88), xycoords="axes fraction", size=11, ha="left", color="red")
plt.text(1.1, 0.85, f"p-value: {p_value:.5f}", transform=plt.gca().transAxes, ha="left", color="red")
plt.text(1.1, 0.82, f"standard error: {std_err:.3f}", transform=plt.gca().transAxes, ha="left", color="red")
# add the r value to the plot as text
ax.text(1.01, 0.92, f"R1 = {r:.3f}", transform=ax.transAxes, fontsize=11)
# fit a linear regression model to the second dataset
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x4, y4)

# calculate the y-values for the fitted line
y_pred2 = slope2 * x4 + intercept2

# plot the second scatter plot and linear regression
# set the edge color of the line
sns.regplot(x=x4, y=y4, color="blue", ax=ax, label="Center", line_kws={"color": "blue", "linewidth": 1.3, "zorder": 15})

# add the equation of the fitted line to the plot
plt.annotate(f"Y2 = {slope2:.4f}X2 + {intercept2:.2f}", xy=(1.01, 0.7), xycoords="axes fraction", size=11, ha="left", color="blue")
# add the p-value and standard error to the plot
plt.text(1.1, 0.67, f"p-value: {p_value2:.5f}", transform=plt.gca().transAxes, ha="left", color="blue")
plt.text(1.1, 0.64, f"standard error: {std_err2:.3f}", transform=plt.gca().transAxes, ha="left", color="blue")
# add the r value to the plot as text
ax.text(1.01, 0.73, f"R2 = {r2:.3f}", transform=ax.transAxes, fontsize=11)
# made  plot
sns.lineplot(x=[x_min, x_max], y=[y_min, y_max], color="gray", linestyle="dashed", linewidth=0.99, label="Bisector of 90"+'\u00b0')
# set the x and y axis limits to include the plot frames
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# add a legend to the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# add the tick labels for the x and y axes
ax.set_yticklabels(range(y_min, y_max+1))
ax.set_xticklabels(range(x_min, x_max+1))
# Show the second plot
plt.show()
plt.show()
#regresion analysis to check it out are they significnt from eachother at given regions between drug and CNTL animals
t_statistic, p_value = stats.ttest_ind(y1, y3)
# Print the t-test results 
#this section is beween corner for druged and CNTL animal behavior
print(f"t-statistic_corner_cntl_VS_treated: {t_statistic}")
print(f"p-value_corner_cntl_VS_treated: {p_value}")
# Perform the F-test
f_statistic, p_value = stats.f_oneway(y1, y3)
print(f"f-statistic_center_cntl_VS_treated: {f_statistic}")
print(f"p-value_center_cntl_VS_treated: {p_value}")
#################################################################center section###############################################
# do the t-test
#this is beween center for both the druged and  the CNTL animal behavior
t_statistic, p_value = stats.ttest_ind(y2, y4)
# Print the t-test results
print(f"t-statistice_center_cntl_VS_treated: {t_statistic}")
print(f"p-valuee_center_cntl_VS_treated: {p_value}")

# do the F-test
f_statistic, p_value = stats.f_oneway(y2, y4)
# print the F-test results 
print(f"f-statistice_center_cntl_VS_treated: {f_statistic}")
print(f"p-value_center_cntl_VS_treated: {p_value}")
