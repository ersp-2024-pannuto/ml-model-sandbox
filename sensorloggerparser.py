import pandas as pd
import os

df1 = pd.read_csv('test_data\Accelerometer.csv')
df2 = pd.read_csv('test_data\Gyroscope.csv')

combined_df = pd.concat([df1, df2], axis=1)
print(combined_df.head(1))
combined_df.to_csv('combined_output.csv', index=False)
# # Set the directory where your CSV files are located
# directory = 'test_data'




# # List of files and the columns you want to keep from each file
# # Format: 'filename.csv': ['column1', 'column2', ...]
# columns_to_keep = {
#     'Accelerometer.csv': ['accelZ', 'accelY', 'accelX'],
#     'Gyroscope.csv': ['gyroZ', 'gyroY', 'gyroX']
# }

# # Desired column order for the final combined CSV
# desired_column_order = ['accelX', 'accelY', 'accelZ', 'gyroX', 'gyroY', 'gyroZ']

# # List to hold all dataframes
# dfs = []

# # Loop through each specified CSV file
# for filename, cols in columns_to_keep.items():
#     filepath = os.path.join(directory, filename)
    
#     if os.path.exists(filepath):
#         # Read only the specific columns from the CSV
       
#         # Ensure all columns in desired_column_order are included (fill missing ones)
#         for col in desired_column_order:
#             if col not in df.columns:
#                 df[col] = pd.NA  # Add missing columns with NaN values
        
#         # Reorder the columns to match the desired order
#         df = df[desired_column_order]
        
#         dfs.append(df)  # Add the DataFrame to the list
#     else:
#         print(f"File {filename} not found!")

# # Concatenate the DataFrames vertically (stacking rows)
# combined_df = pd.concat(dfs, ignore_index=True)

# # Fill missing values across rows
# combined_df = combined_df.fillna(method='ffill', axis=0)  # Forward fill missing values

# # Save the combined dataframe to a new CSV file
# combined_df.to_csv('combined_output.csv', index=False)

# print("CSV files with specific columns and merged rows (no missing values) have been combined successfully!")


# #df pandas.join using matching with seconds elapsed