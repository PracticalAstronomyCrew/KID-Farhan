#!/usr/bin/env python
# coding: utf-8

# In[1]:


locations = ["Erica Winter", "Erica Summer"]

#35 items
# linestyles = ['-', '--', '-.', ':']
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'teal']
# unique = []
# for i, location, in enumerate(locations):
#     linestyle = linestyles[i % len(linestyles)]
#     color = colors[i % len(colors)]
#     unique.append((linestyle,color))
# if len(unique) == len(set(unique)):
#     print("No duplicates in the list.")
# else:
#     print("Duplicates found in the list.")
    
# print(unique)


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import ephem
from datetime import datetime

# Algorithm for finding sun and moon altitudes. a, b, c, d, and e are year, month, day, hour, and minute respectively.
def hoogte(a, b, c, d, e):
    home = ephem.Observer()
    home.lat, home.lon = "52.695", "6.896"
    try:
        home.date = datetime(a, b, c, d, e, 0)
    except ValueError:
        return None, None

    sun = ephem.Sun()
    sun.compute(home)
    moon = ephem.Moon()
    moon.compute(home)
    return round(float(sun.alt) * 57.2957795), round(float(moon.alt) * 57.2957795)


# In[3]:


import os
import pandas as pd
import ephem
from datetime import datetime

# Define the filtering function
def filter_sun_moon(df):
    filtered_rows = []
    for index, row in df.iterrows():
        year, month, day, hour, minute = (
            int(row["Year"]),
            int(row["Month"]),
            int(row["Day"]),
            int(row["Hour"]),
            int(row["Minute"]),
        )
        
        # Calculate sun and moon altitudes using the hoogte function
        sun_altitude, moon_altitude = hoogte(year, month, day, hour, minute)
        
        # Check if both sun and moon altitudes are above -18 degrees
        if sun_altitude is not None and moon_altitude is not None:
            if sun_altitude != 500 and moon_altitude != -500:
                filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)

lists = {}  # Create a dictionary to store lists

for j in locations:
    # Specify the folder containing your data files
    folder_path = r"C:\Users\Farhan\Desktop\\" + str(j)

    # Create an empty list to store DataFrames
    list_name = f"dfs{j}"
    lists[list_name] = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".dat"):  # Check if the file is a .dat file (or adjust the extension as needed)
            file_path = os.path.join(folder_path, filename)

            # Read the current data file into a DataFrame and append it to the list
            df = pd.read_csv(file_path, delimiter=";", skiprows=35,
                             names=["Date and Time", "Date and Time2", "Temperature", "Number", "Hz", "Magnitude"])
            lists[list_name].append(df)

    for i, df in enumerate(lists[list_name]):
        lists[list_name][i] = df.drop(columns=["Date and Time2", "Temperature", "Number", "Hz"])  # Dropping these columns from dat files
        lists[list_name][i][["Date", "Time"]] = df["Date and Time"].str.split("T", expand=True)  # Separating date and time (local)

        # Removing rows where Magnitude is zero
        lists[list_name][i] = lists[list_name][i][lists[list_name][i]["Magnitude"] != -0.0]

        lists[list_name][i][["Year", "Month", "Day"]] = lists[list_name][i]["Date"].str.split("-", expand=True)  # Splitting date column
        lists[list_name][i] = lists[list_name][i].drop(columns=["Date", "Date and Time"])  # Dropping these columns from data

        lists[list_name][i][["Hour", "Minute", "Second"]] = lists[list_name][i]["Time"].str.split(":", expand=True)  # Splitting time column

        lists[list_name][i] = lists[list_name][i][["Year", "Month", "Day", "Hour", "Minute", "Magnitude"]]  # Sorting

        # Apply the filtering function to remove rows with sun and moon below -18 degrees
        lists[list_name][i] = filter_sun_moon(lists[list_name][i])


# In[6]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))
all_data = pd.DataFrame(columns=['Location', 'Time', 'Magnitude'])
linestyles = ['-', '--', '-.', ':']
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange', 'teal']

for i, location, in enumerate(locations):
    # Filter out empty DataFrames from the list
    dfs_location = [df for df in lists[f'dfs{location}'] if not df.empty]

    # Combine the 'Hour' and 'Minute' columns into a single 'Time' column
    for df in dfs_location:
        df['Hour'] = df['Hour'].astype(int).apply(lambda hour: hour - 24 if hour > 9 else hour)
        df['Time'] = df['Hour'] + df['Minute'].astype(int) / 60.0

    combined_data_location = pd.concat(dfs_location, ignore_index=True)

    # Extract the 'Time' and 'Magnitude' columns from the combined data
    time_column_location = combined_data_location['Time']
    magnitude_column_location = combined_data_location['Magnitude']

    # Group data by time intervals and calculate average magnitude
    time_intervals_location = pd.cut(time_column_location, bins=np.linspace(-10, 9, num=500))
    average_magnitude_location = combined_data_location.groupby(time_intervals_location, observed=False)['Magnitude'].quantile(0.9)

    # Extract midpoints of time intervals
    midpoints_location = [interval.mid for interval in average_magnitude_location.index.categories]

    linestyle = linestyles[i % len(linestyles)]
    color = colors[i % len(colors)]
    ax.plot(midpoints_location, average_magnitude_location.values, linestyle=linestyle, color=color, label=location)
    
    # Add data for the current location to the all_data DataFrame
    location_data = pd.DataFrame({'Location': [location] * len(midpoints_location),
                                 'Time': midpoints_location,
                                 'Magnitude': average_magnitude_location.values})
    all_data = pd.concat([all_data, location_data], ignore_index=True)

    
ax.set_title("Combined Jellyfish Plot", fontsize=15)
ax.set_ylabel("NSB (mag/arcsec^2)", fontsize=14)
ax.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6])
ax.set_xticklabels(["16:00", "18:00", "20:00", "22:00", "00:00", "02:00", "04:00", "06:00"])
ax.set_xlim(-8, 8)
ax.set_ylim(16, 23)
ax.set_xlabel("Time of the day (hours and minutes)", fontsize=14)
ax.grid(True)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

output_filename = 'erica_divided.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"PNG file '{output_filename}' saved successfully.")
# Show the plot
plt.tight_layout()
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, ax = plt.subplots(figsize=(12, 6))
all_data = pd.DataFrame(columns=['Location', 'Time', 'Magnitude'])
linestyles = ['-', '--', '-.', ':']
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'orange', 'teal']

for i, location, in enumerate(locations):
    # Filter out empty DataFrames from the list
    dfs_location = [df for df in lists[f'dfs{location}'] if not df.empty]

    # Combine the 'Hour' and 'Minute' columns into a single 'Time' column
    for df in dfs_location:
        df['Hour'] = df['Hour'].astype(int).apply(lambda hour: hour - 24 if hour > 9 else hour)
        df['Time'] = df['Hour'] + df['Minute'].astype(int) / 60.0

    combined_data_location = pd.concat(dfs_location, ignore_index=True)

    # Extract the 'Time' and 'Magnitude' columns from the combined data
    time_column_location = combined_data_location['Time']
    magnitude_column_location = combined_data_location['Magnitude']

    # Group data by time intervals and calculate average magnitude
    time_intervals_location = pd.cut(time_column_location, bins=np.linspace(-10, 9, num=500))
    average_magnitude_location = combined_data_location.groupby(time_intervals_location, observed=False)['Magnitude'].quantile(0.9)

    # Extract midpoints of time intervals
    midpoints_location = [interval.mid for interval in average_magnitude_location.index.categories]

    linestyle = linestyles[i % len(linestyles)]
    color = colors[i % len(colors)]
    ax.plot(midpoints_location, average_magnitude_location.values, linestyle=linestyle, color=color, label=location)
    
    # Add data for the current location to the all_data DataFrame
    location_data = pd.DataFrame({'Location': [location] * len(midpoints_location),
                                 'Time': midpoints_location,
                                 'Magnitude': average_magnitude_location.values})
    all_data = pd.concat([all_data, location_data], ignore_index=True)

    
excel_filename = 'combined_data.xlsx'
all_data.to_excel(excel_filename, index=False)
print(f"Excel file '{excel_filename}' saved successfully.")
    
ax.set_title("Combined Jellyfish Plot", fontsize=15)
ax.set_ylabel("NSB (mag/arcsec^2)", fontsize=14)
ax.set_xticks([-8, -6, -4, -2, 0, 2, 4, 6])
ax.set_xticklabels(["16:00", "18:00", "20:00", "22:00", "00:00", "02:00", "04:00", "06:00"])
ax.set_xlim(-8, 8)
ax.set_ylim(16, 23)
ax.set_xlabel("Time of the day (hours and minutes)", fontsize=14)
ax.grid(True)

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

output_filename = 'average_heatmaps.png'
#plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#print(f"PNG file '{output_filename}' saved successfully.")
# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:




