import pandas as pd
from pathlib import Path
import pickle
import re

from data.preprocessing_utils import vertical_split, horizontal_split, categorical_values_to_int


def preprocess_adult_income_dataset() -> pd.DataFrame:
    df = pd.read_csv('income/adult.csv')
    # replace ? as unknown. seems really stupid, but will put the code here anyway.
    for col in df.columns:
        if "?" in df[col].unique().tolist():
            df[col] = df[col].apply(lambda x: 'unknown' if x == '?' else x)

    cat_cols = [
        'workclass', 'educational-num', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'native-country'
    ]
    num_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    label_col = 'income'

    df.reset_index(drop=True, inplace=True)
    ids = ['N%05d' % i for i in range(len(df))]
    df.insert(0, 'ID', ids)

    df = df.loc[:, ['ID'] + cat_cols + num_cols + [label_col]]
    df.set_index('ID', inplace=True)
    for col in cat_cols + [label_col]:
        lst = sorted(df[col].unique().tolist())
        mapping = dict(zip(lst, range(len(lst))))
        df[col] = df[col].apply(lambda x: mapping[x])
    for col in num_cols:
        arr = df[col].values
        df[col] = (arr - arr.min()) / (arr.max() - arr.min())
    return df


def partition(full_df, dir_pth, col_sets, num_partitions):
    # drop the 'duration' column
    sub_dfs = vertical_split(full_df, col_sets)
    # dataframe of Active Party
    df_ap = sub_dfs[0]
    # create partitions for passive parties
    df_pp_lst = []
    for n, sub_df in zip(num_partitions, sub_dfs[1:]):
        df_pp_lst += horizontal_split(sub_df, n, ordered=True)

    # save dfs
    df_ap.to_csv(dir_pth / 'p0_data.csv', index=False)
    for i, df_pp in enumerate(df_pp_lst):
        df_pp.to_csv(dir_pth / f'p{i + 1}_data.csv', index=False)


def main():
    # df = preprocess_adult_income_dataset()
    # df.to_csv('income/preprocessed.csv', index=True)
    dir_pth = Path('income')
    df = pd.read_csv(dir_pth / 'preprocessed.csv')
    cols = [
        ['ID', 'workclass', 'occupation', 'capital-gain', 'capital-loss', 'hours-per-week', 'income'],
        ['ID', 'marital-status', 'relationship', 'race', 'gender', 'native-country', 'age'],
        ['ID', 'educational-num']
    ]
    partition(df, dir_pth, cols, [2, 2])


if __name__ == '__main__':
    main()
