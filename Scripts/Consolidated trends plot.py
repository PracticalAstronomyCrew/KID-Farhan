# Define locations with their coordinates (not complete list)
locations_dict = {
    "Ameland-Natuurcentrum-Nes": [53.449, 5.775],
    "Boerakker": [53.187, 6.329],
    "Ostland": [53.607, 6.727],
}


# Select a location from the dictionary
selectedlocation = list(locations_dict.keys())[31]
print(selectedlocation)

import os
import pandas as pd
import ephem
from datetime import datetime

# Load the CSV file into a DataFrame
file_path = f'{selectedlocation} cloudy or not.csv'
result_df = pd.read_csv(file_path)

# Extract date and time components from filename
def extract_datetime_components(df):
    df['year'] = df['filename'].str.slice(28, 32)
    df['month'] = df['filename'].str.slice(32, 34)
    df['day'] = df['filename'].str.slice(34, 36)
    df['hour'] = df['filename'].str.slice(36, 38)
    df['minute'] = df['filename'].str.slice(38, 40)
    return df

result_df = extract_datetime_components(result_df)

# Filter for cloudy conditions at specific hours
def filter_cloudy_conditions(df, hours_and_minutes):
    filtered_dfs = []
    for hour, minute in hours_and_minutes:
        filtered_df = df[(df['datavals'] == 'Cloudy') & ((df['hour'] == hour) | ((df['hour'] == hour) & (df['minute'] == minute)))]
        filtered_df['date'] = pd.to_datetime(df[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1))
        date_list = filtered_df['date'].dt.strftime('%Y-%m-%d').tolist()
        filtered_dfs.append(date_list)
    return filtered_dfs

hours_and_minutes_list = [
    ('21', '00'),
    ('22', '00'),
    ('23', '00'),
    ('00', '00'),
    ('01', '00'),
    ('02', '00')
]
date_lists = filter_cloudy_conditions(result_df, hours_and_minutes_list)

# Define the filtering function for sun and moon altitudes and moon phase
def hoogte(a, b, c, d, e):
    home = ephem.Observer()
    home.lat, home.lon = "36.824167", "30.335556"
    try:
        home.date = datetime(a, b, c, d, e, 0)
    except ValueError:
        return None, None
    sun = ephem.Sun()
    sun.compute(home)
    moon = ephem.Moon()
    moon.compute(home)
    moon_phase = ephem.Moon(home)
    illuminated_fraction = moon_phase.moon_phase
    moon_phase_percent = illuminated_fraction * 100
    return round(float(sun.alt) * 57.2957795), round(float(moon.alt) * 57.2957795), round(moon_phase_percent,2)

# Define the function to filter sun and moon altitudes
def filter_sun_moon(df):
    filtered_rows = []
    for index, row in df.iterrows():
        year, month, day, hour, minute = int(row["Year"]), int(row["Month"]), int(row["Day"]), int(row["Hour"]), int(row["Minute"])
        sun_altitude, moon_altitude = hoogte(year, month, day, hour, minute)
        if sun_altitude is not None and moon_altitude is not None:
            if sun_altitude < -18:
                if 0 <= moon_altitude <= 5:
                    moon_phase_threshold = 50 - (8*moon_altitude)
                    if moon_phase < moon_phase_threshold:
                        filtered_rows.append(row)
                if -3 <= moon_altitude < 0:
                    moon_phase_threshold = (150-(50*moon_altitude))/3
                    if moon_phase < moon_phase_threshold:
                        filtered_rows.append(row)
                if moon_altitude < -3:
                    filtered_rows.append(row)

# Process .dat files and filter based on sun and moon altitudes
folder_path = r"C:\Users\Farhan\Desktop\SQM Data\\" + selectedlocation
dfs = []

for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, delimiter=";", skiprows=35,
                         names=["Date and Time", "Date and Time2", "Temperature", "Number", "Hz", "Magnitude"]
                        )
        dfs.append(df)

# Clean and process each DataFrame
for i, df in enumerate(dfs):
    dfs[i] = df.drop(columns=["Date and Time2", "Temperature", "Number", "Hz"])
    dfs[i][["Date","Time"]] = df["Date and Time"].str.split("T", expand=True)
    dfs[i] = dfs[i][dfs[i]["Magnitude"] != -0.0]
    dfs[i][["Year","Month","Day"]] = dfs[i]["Date"].str.split("-", expand=True)
    dfs[i] = dfs[i].drop(columns=["Date", "Date and Time"])
    dfs[i][["Hour","Minute","Second"]] = dfs[i]["Time"].str.split(":", expand=True)
    dfs[i] = dfs[i][["Year","Month","Day","Hour","Minute","Magnitude"]]
    dfs[i] = filter_sun_moon(dfs[i])

# Filter out hours with insufficient data
def filter_variance_and_slope(df):
    if len(df) == 0:
        return df
    filtered_hours = [hour_df for hour, hour_df in df.groupby("Hour") if len(hour_df) >= 30]
    if len(filtered_hours) == 0:
        return pd.DataFrame()
    return pd.concat(filtered_hours)

for i, df in enumerate(dfs):
    dfs[i] = filter_variance_and_slope(df)

# Plotting and analysis
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

plt.figure(figsize=(12, 6))
hours = [21,22,23,0,1,2]
finallist = []

for specified_hour in hours:
    df = pd.concat(dfs)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])

    excluded_dates = date_lists[hours.index(specified_hour)]
    df = df[~df['Datetime'].dt.strftime('%Y-%m-%d').isin(excluded_dates)]

    df['Linear_Magnitude'] = 1.08 * (10**8) * 10**(-0.4 * df['Magnitude'])
    df.loc[df['Linear_Magnitude'] > 10, 'Linear_Magnitude'] = None
    df.dropna(subset=['Linear_Magnitude'], inplace=True)

    hourly_mean = df[df['Datetime'].dt.hour == specified_hour].resample('H', on='Datetime')['Linear_Magnitude'].median()
    min_date = df[df['Datetime'].dt.hour == specified_hour]['Datetime'].min()
    max_date = df[df['Datetime'].dt.hour == specified_hour]['Datetime'].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='H')
    hourly_std = df[df['Datetime'].dt.hour == specified_hour].resample('H', on='Datetime')['Linear_Magnitude'].std()

    magnitude_std_values = hourly_std.values
    magnitude_std_with_gaps = np.empty(len(date_range))
    magnitude_std_with_gaps[:] = np.nan
    magnitude_std_with_gaps[:len(magnitude_std_values)] = magnitude_std_values

    plt.errorbar(
        date_range, hourly_mean, yerr=magnitude_std_with_gaps, fmt='o', color='darkblue', label='Median Linear Magnitude'
    )

    non_nan_indices = ~np.isnan(hourly_mean)
    X = np.arange(len(date_range))[non_nan_indices]
    y = hourly_mean[non_nan_indices]

    reg = LinearRegression().fit(X.reshape(-1, 1), y)
    trend = reg.predict(np.arange(len(date_range)).reshape(-1, 1))

    plt.plot(
        date_range[non_nan_indices], trend[non_nan_indices], linestyle='--', color='red', label=f'Trend Line'
    )

    plt.ylim(0,2)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.xlabel('Time',fontsize=15)
    plt.ylabel(r'Median Linear Magnitude (mcd/m$^2$)',fontsize=15)
    plt.title(f'Median Linear Magnitude at {specified_hour}:00 | Groningen-ZernikeCampus',fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.show()

    slope_per_hour = reg.coef_[0]
    slope_per_year = slope_per_hour * 365.25 * 24
    trend_direction = "increasing" if slope_per_year > 0 else "decreasing" if slope_per_year < 0 else "constant"
    percentage_change_per_year = (slope_per_year / hourly_mean.head(6).median()) * 100

    print(f"The overall trend is {trend_direction}.")
    print(f"The slope of the trend line is {slope_per_year:.6f} mcd/m2 per year.")
    print(f"The percentage change per year is {percentage_change_per_year:.6f}%.")

    finallist.extend([str(specified_hour), str(slope_per_year), str(percentage_change_per_year)])

print(finallist)
