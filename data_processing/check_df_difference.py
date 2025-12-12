import pandas as pd


def df_difference(df1, df2):
    # Find and print different columns
    diff_columns = df1.columns.difference(df2.columns).tolist() + df2.columns.difference(df1.columns).tolist()
    if diff_columns:
        print(f"Different columns: {diff_columns}")
    else:
        print("No different columns.")



if __name__ == "__main__":
    df1 = pd.read_csv("data/players_data/merged_gw_21_22.csv", engine="python")
    df2 = pd.read_csv("data/players_data/merged_gw_25_26.csv", engine="python")
    df_difference(df1, df2)