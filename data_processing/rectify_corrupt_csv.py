import pandas as pd

# The function below aims to rectify the csv files if there is a mismatch in
# the dimension of the rows


def rectify_corrupt_csv(file_path: str):

    try:
        # Read ONLY the header row (nrows=0) to get column names efficiently
        print(f"Fetching headers from the top of: {file_path}")
        correct_headers = pd.read_csv(file_path, nrows=0).columns.tolist()
        print(f"Successfully loaded {len(correct_headers)} headers.")

    except Exception as e:
        print(f"Could not read headers from the file: {e}")
        return None

    max_columns = 49
    column_names = [f"col_{i}" for i in range(max_columns)]

    try:
        # Try reading with error handling for malformed lines
        df = pd.read_csv(
            file_path, header=None, names=column_names, skiprows=1, engine="python"
        )
        print("File found with columns: ")
        print(df.columns.tolist())
        print(f"Shape: {df.shape}")

    except FileNotFoundError:
        print("Error: 'merged_gw.csv' not found.")
        exit()

    bad_rows_filter = df["col_42"].notna()

    # Seperate the dataset into two parts (the 42 and 49 column row variants)
    bad_rows = df[bad_rows_filter].copy()
    good_rows = df[~bad_rows_filter].copy()

    # Drop bad columns and save to new variable
    cols_to_drop_bad = [f"col_{i}" for i in range(22, 29)]
    bad_rows_cleaned = bad_rows.drop(columns=cols_to_drop_bad)

    cols_to_drop_good = [f"col_{i}" for i in range(42, 49)]
    good_rows_cleaned = good_rows.drop(columns=cols_to_drop_good)

    # Match the column names
    bad_rows_cleaned.columns = good_rows_cleaned.columns

    # 5. Combine the two cleaned DataFrames
    df_final = pd.concat([good_rows_cleaned, bad_rows_cleaned], ignore_index=True)

    if len(correct_headers) == len(df_final.columns):
        df_final.columns = correct_headers
        print(f"✅ Successfully cleaned data. Final shape: {df_final.shape}")
    else:
        print(
            f"Error: Header count ({len(correct_headers)}) does not match column count ({len(df_final.columns)})."
        )
        return None

    return df_final


clean_data = rectify_corrupt_csv(
    "/Users/ola/Documents/FPL_Price_Predictor/raw_corrupt_merged_gw_24_25.csv"
)

if clean_data is not None:
    print("\n--- Columns of the Final DataFrame ---")
    print(clean_data.columns.tolist())

    # --- How to save the file ---
    output_filename = "cleaned_fpl_data.csv"
    clean_data.to_csv(output_filename, index=False)

    print(f"\n✅ DataFrame successfully saved to '{output_filename}'")
