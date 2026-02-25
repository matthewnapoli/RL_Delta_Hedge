import pandas as pd

GroupedData = tuple[tuple, pd.DataFrame]

def preprocess_kaggle_data(df: pd.DataFrame) -> tuple[list[GroupedData], list[GroupedData]]:
    df["[EXPIRE_DATE]"] = pd.to_datetime(df["[EXPIRE_DATE]"])
    df["[QUOTE_DATE]"] = pd.to_datetime(df["[QUOTE_DATE]"])
    df = df.sort_values("[QUOTE_DATE]")
    
    df_train = df[df["[QUOTE_DATE]"] < "2022-06-10"]
    df_test  = df[df["[QUOTE_DATE]"] >= "2022-06-10"]
    
    groups_train_sorted = sorted(
        (g for g in df_train.groupby(["[EXPIRE_DATE]", "[STRIKE]"]) if len(g[1]) > 25),
        key=lambda g: g[1]["[QUOTE_DATE]"].max()
    )
    
    groups_test_sorted = sorted(
        (g for g in df_test.groupby(["[EXPIRE_DATE]", "[STRIKE]"]) if len(g[1]) > 25),
        key=lambda g: g[1]["[QUOTE_DATE]"].max()
    )
    
    return groups_train_sorted, groups_test_sorted