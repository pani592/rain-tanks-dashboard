# 1, 1b - Clean up Whenuapai Rainfall data
import pandas as pd

headers = ["DateAndTime", "Rainfall_mm", "Qual", "Comments"]
df = pd.read_excel('raw_data/HYCSV_Hourly_Rainfall_Whenuapai.xlsx', header=None, skiprows=4, names=headers)
df = df.drop(columns=["Qual", "Comments"])
df['Rainfall_mm'] = df['Rainfall_mm'].fillna(0).astype(float)
df['DateAndTime'] = pd.to_datetime(df['DateAndTime'])
df = df.resample('D', on='DateAndTime').sum().reset_index(drop=False) # daily sums for rainfall
df2 = df.loc[df.Rainfall_mm > 0, :]
df.to_parquet("data/transformed_data/daily_rainfall.parquet") 
