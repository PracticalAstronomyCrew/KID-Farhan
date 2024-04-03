#!/usr/bin/env python
# coding: utf-8

# In[268]:


locations_dict = {
    "Ameland-Natuurcentrum-Nes": [53.449, 5.775],
    "Boerakker": [53.187, 6.329],
    "Ostland": [53.607, 6.727],
    "Borkum": [53.587, 6.663],
    "DeZilk": [52.301, 4.542],
    "Gorredijk": [53.008, 6.074],
    "DeHeld": [53.228, 6.512],
    "Zernike": [53.24, 6.536],
    "Haaksbergen": [52.149, 6.718],
    "Heerenveen01": [52.967, 5.94],
    "Heerenveen-Station": [52.96, 5.915],
    "Hippolytushoef": [52.929, 4.986],
    "Hornhuizen": [53.403, 6.352],
    "Lauwersoog": [53.385, 6.235],
    "Leiden": [52.155, 4.483],
    "Lochem": [52.172, 6.401],
    "Oostkapelle": [51.572, 3.537],
    "Rijswijk": [52.026, 4.314],
    "Roodeschool": [53.412, 6.755],
    "Sellingen": [52.938, 7.131],
    "Texel": [53.003, 4.787],
    "Vlieland-Oost": [53.295, 5.092],
    "Weerribben": [52.788, 5.938],
    "Westhoek": [53.272, 5.558],
    "ZwarteHaan": [53.309, 5.628],
    "Oldenburg": [53.15292605, 8.165231368],
    "tZandt": [53.3695, 6.7712],
    "Tolbert": [53.1651, 6.3596],
    "Noordpolderzijl": [53.4325, 6.5837],
    "Moddergat": [53.40656, 6.06985],
    "Delft": [51.99009, 4.37541],
    "Akkrum": [53.049374, 5.834147]
}


selectedlocation = list(locations_dict.keys())[31]
print(selectedlocation)


# In[269]:


import os
import pygrib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Specify the file path
file_path = f'{selectedlocation} cloudy or not.csv'

# Read the CSV file into a DataFrame
result_df = pd.read_csv(file_path)
# Now 'result_df' contains the data from the CSV file


# In[270]:


import pandas
pandas.set_option('display.max_rows', None)

df = result_df
df['year'] = df['filename'].str.slice(28, 32)
df['month'] = df['filename'].str.slice(32, 34)
df['day'] = df['filename'].str.slice(34, 36)
df['hour'] = df['filename'].str.slice(36, 38)
df['minute'] = df['filename'].str.slice(38, 40)
df
#filtered_df = df[(df['datavals'] == 'Cloudy') & (df['hour'] == '21')].drop(columns='filename')
filtered_df = df[(df['datavals'] == 'Cloudy') & (((df['hour'] == '21') | ((df['hour'] == '22') & (df['minute'] == '00'))))].drop(columns='filename')
filtered_df
filtered_df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
filtered_df
# Convert the 'date' column to a list
date_list = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()


df = result_df
df['year'] = df['filename'].str.slice(28, 32)
df['month'] = df['filename'].str.slice(32, 34)
df['day'] = df['filename'].str.slice(34, 36)
df['hour'] = df['filename'].str.slice(36, 38)
df['minute'] = df['filename'].str.slice(38, 40)
df
#filtered_df = df[(df['datavals'] == 'Cloudy') & (df['hour'] == '21')].drop(columns='filename')
filtered_df = df[(df['datavals'] == 'Cloudy') & (((df['hour'] == '22') | ((df['hour'] == '23') & (df['minute'] == '00'))))].drop(columns='filename')
filtered_df
filtered_df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
filtered_df
# Convert the 'date' column to a list
date_list2 = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()


df = result_df
df['year'] = df['filename'].str.slice(28, 32)
df['month'] = df['filename'].str.slice(32, 34)
df['day'] = df['filename'].str.slice(34, 36)
df['hour'] = df['filename'].str.slice(36, 38)
df['minute'] = df['filename'].str.slice(38, 40)
df
#filtered_df = df[(df['datavals'] == 'Cloudy') & (df['hour'] == '21')].drop(columns='filename')
filtered_df = df[(df['datavals'] == 'Cloudy') & (((df['hour'] == '23') | ((df['hour'] == '00') & (df['minute'] == '00'))))].drop(columns='filename')
filtered_df
filtered_df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
filtered_df
# Convert the 'date' column to a list
date_list3 = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()


df = result_df
df['year'] = df['filename'].str.slice(28, 32)
df['month'] = df['filename'].str.slice(32, 34)
df['day'] = df['filename'].str.slice(34, 36)
df['hour'] = df['filename'].str.slice(36, 38)
df['minute'] = df['filename'].str.slice(38, 40)
df
#filtered_df = df[(df['datavals'] == 'Cloudy') & (df['hour'] == '21')].drop(columns='filename')
filtered_df = df[(df['datavals'] == 'Cloudy') & (((df['hour'] == '00') | ((df['hour'] == '01') & (df['minute'] == '00'))))].drop(columns='filename')
filtered_df
filtered_df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
filtered_df
# Convert the 'date' column to a list
date_list4 = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()


df = result_df
df['year'] = df['filename'].str.slice(28, 32)
df['month'] = df['filename'].str.slice(32, 34)
df['day'] = df['filename'].str.slice(34, 36)
df['hour'] = df['filename'].str.slice(36, 38)
df['minute'] = df['filename'].str.slice(38, 40)
df
#filtered_df = df[(df['datavals'] == 'Cloudy') & (df['hour'] == '21')].drop(columns='filename')
filtered_df = df[(df['datavals'] == 'Cloudy') & (((df['hour'] == '01') | ((df['hour'] == '02') & (df['minute'] == '00'))))].drop(columns='filename')
filtered_df
filtered_df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
filtered_df
# Convert the 'date' column to a list
date_list5 = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()


df = result_df
df['year'] = df['filename'].str.slice(28, 32)
df['month'] = df['filename'].str.slice(32, 34)
df['day'] = df['filename'].str.slice(34, 36)
df['hour'] = df['filename'].str.slice(36, 38)
df['minute'] = df['filename'].str.slice(38, 40)
df
#filtered_df = df[(df['datavals'] == 'Cloudy') & (df['hour'] == '21')].drop(columns='filename')
filtered_df = df[(df['datavals'] == 'Cloudy') & (((df['hour'] == '02') | ((df['hour'] == '03') & (df['minute'] == '00'))))].drop(columns='filename')
filtered_df
filtered_df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
filtered_df
# Convert the 'date' column to a list
date_list6 = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()




# In[271]:


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
    home.lat, home.lon = str(locations_dict[f"{selectedlocation}"][0]), str(locations_dict[f"{selectedlocation}"][1])
    try:
        home.date = datetime(a, b, c, d, e, 0)
    except ValueError:
        return None, None
    sun = ephem.Sun()
    sun.compute(home)
    moon = ephem.Moon()
    moon.compute(home)
    return round(float(sun.alt) * 57.2957795), round(float(moon.alt) * 57.2957795)


# In[272]:


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
            if sun_altitude < -18 and moon_altitude < -3:
                filtered_rows.append(row)
    
    return pd.DataFrame(filtered_rows)

# Specify the folder containing your data files
folder_path = r"C:\Users\Farhan\Desktop\SQM Data\\" + selectedlocation

# Create an empty list to store DataFrames
dfs = []

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):  # Check if the file is a .dat file (or adjust the extension as needed)
        file_path = os.path.join(folder_path, filename)
        
        # Read the current data file into a DataFrame and append it to the list
        df = pd.read_csv(file_path, delimiter=";", skiprows=35,
                         names=["Date and Time", "Date and Time2", "Temperature", "Number", "Hz", "Magnitude"])
        dfs.append(df)
        
for i, df in enumerate(dfs):
    dfs[i] = df.drop(columns=["Date and Time2", "Temperature", "Number", "Hz"])  # Dropping these columns from dat files
    dfs[i][["Date","Time"]] = df["Date and Time"].str.split("T", expand=True)  # Separating date and time (local)
    
    # Removing rows where Magnitude is zero
    dfs[i] = dfs[i][dfs[i]["Magnitude"] != -0.0]
    
    dfs[i][["Year","Month","Day"]] = dfs[i]["Date"].str.split("-", expand=True)  # Splitting date column
    dfs[i] = dfs[i].drop(columns=["Date", "Date and Time"])  # Dropping these columns from data
    
    dfs[i][["Hour","Minute","Second"]] = dfs[i]["Time"].str.split(":", expand=True)  # Splitting time column
    
    dfs[i] = dfs[i][["Year","Month","Day","Hour","Minute","Magnitude"]]  # Sorting
    
    # Apply the filtering function to remove rows with sun and moon below -18 degrees
    dfs[i] = filter_sun_moon(dfs[i])


# In[273]:


import pandas as pd
import numpy as np

# Define the additional filtering function
def filter_variance_and_slope(df):
    if len(df) == 0:
        return df  # Return the empty DataFrame as is if it's empty

    filtered_hours = []
    for hour, hour_df in df.groupby("Hour"):
        if len(hour_df) >= 30:
            filtered_hours.append(hour_df)

    if len(filtered_hours) == 0:
        return pd.DataFrame()  # Return an empty DataFrame if all hours were filtered out

    return pd.concat(filtered_hours)

# Apply the additional filtering function to remove hours with variance > 0.004, slope not between -0.002 and 0.002, and less than 30 values
for i, df in enumerate(dfs):
    dfs[i] = filter_variance_and_slope(df)


# In[274]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


plt.figure(figsize=(12, 6))
hours = [21,22,23,0,1,2]
finallist = []
for specified_hour in hours:
    # Specify the hour you want to plot (e.g., 12 for noon)

    # Concatenate the DataFrames into a single DataFrame
    df = pd.concat(dfs)

    # Convert 'Year', 'Month', 'Day', 'Hour', and 'Minute' to datetime
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])

    # Specify the date(s) you want to exclude
    if specified_hour == 21:
        excluded_dates = date_list
    elif specified_hour == 22:
        excluded_dates = date_list2
    elif specified_hour == 23:
        excluded_dates = date_list3
    elif specified_hour == 0:
        excluded_dates = date_list4
    elif specified_hour == 1:
        excluded_dates = date_list5
    elif specified_hour == 2:
        excluded_dates = date_list6

    # Filter out the excluded date(s) from the DataFrame
    df = df[~df['Datetime'].dt.strftime('%Y-%m-%d').isin(excluded_dates)]

    # Convert magnitude values to linear scale
    df['Linear_Magnitude'] = 1.08 * (10**8) * 10**(-0.4 * df['Magnitude'])

    #CHECK THIS VALUE 



    df.loc[df['Linear_Magnitude'] > 10, 'Linear_Magnitude'] = None



    #CHECK THIS VALUE

    df.dropna(subset=['Linear_Magnitude'], inplace=True)

    # Group by Datetime and calculate the median linear magnitude for the specified hour
    hourly_mean = df[df['Datetime'].dt.hour == specified_hour].resample('H', on='Datetime')['Linear_Magnitude'].median()

    # Create a continuous time scale with gaps
    min_date = df[df['Datetime'].dt.hour == specified_hour]['Datetime'].min()
    max_date = df[df['Datetime'].dt.hour == specified_hour]['Datetime'].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='H')

    # Calculate standard deviation for each data point
    hourly_std = df[df['Datetime'].dt.hour == specified_hour].resample('H', on='Datetime')['Linear_Magnitude'].std()

    # Create an array with NaN values for the gaps
    magnitude_std_values = hourly_std.values
    magnitude_std_with_gaps = np.empty(len(date_range))
    magnitude_std_with_gaps[:] = np.nan
    magnitude_std_with_gaps[:len(magnitude_std_values)] = magnitude_std_values

    # Plot the data with error bars
    plt.errorbar(date_range, hourly_mean, yerr=magnitude_std_with_gaps, fmt='o', color='darkblue', label='Median Linear Magnitude')

    non_nan_indices = ~np.isnan(hourly_mean)
    X = np.arange(len(date_range))[non_nan_indices]
    y = hourly_mean[non_nan_indices]


    # Fit polynomial regression to the available data points within 1 standard deviation
    degree = 1  # You can adjust the degree of the polynomial as needed
    coefficients = np.polyfit(X, y, degree)
    trend = np.polyval(coefficients, np.arange(len(date_range)))

    # Plot the trend line for the data within 1 standard deviation
    plt.plot(date_range[non_nan_indices], trend[non_nan_indices], linestyle='--', color='red', label=f'Trend Line (degree={degree})')

    non_nan_indices = ~np.isnan(hourly_mean)
    X = np.arange(len(date_range))[non_nan_indices]
    y = hourly_mean[non_nan_indices]

    reg = LinearRegression().fit(X.reshape(-1, 1), y)
    trend = reg.predict(np.arange(len(date_range)).reshape(-1, 1))



    plt.ylim(0,2)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.xlabel('Time',fontsize=15)
    plt.ylabel(r'Median Linear Magnitude (mcd/m$^2$)',fontsize=15)
    plt.title(f'Median Linear Magnitude at 21:00 | Groningen-ZernikeCampus',fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()


    #Check the slope of the trend line to determine the trend direction
    slope_per_hour = reg.coef_[0]
    slope_per_year = slope_per_hour * 365.25 * 24  # Assuming 365.25 days per year

    if slope_per_year > 0:
        trend_direction = "increasing"
    elif slope_per_year < 0:
        trend_direction = "decreasing"
    else:
        trend_direction = "constant"

    print(f"The overall trend is {trend_direction}.")
    print(f"The slope of the trend line is {slope_per_year:.6f} mcd/m2 per year.")

    hourly_mean = hourly_mean.dropna()
    num_initial_values = 6  # You can adjust this number as needed

    # Calculate the median of the initial values
    initial_median = hourly_mean.head(num_initial_values).median()
    print(hourly_mean.head(num_initial_values))

    percentage_change_per_year = (slope_per_year / initial_median) * 100

    print(f"The percentage change per year is {percentage_change_per_year:.6f}%.")

    finallist.append(str(specified_hour))
    finallist.append(str(slope_per_year))
    finallist.append(str(percentage_change_per_year))
    
print(finallist)

