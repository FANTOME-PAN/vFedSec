import pandas as pd
from pathlib import Path
import pickle
import re

from data.preprocessing_utils import vertical_split, horizontal_split, categorical_values_to_int


def preprocess_bank_dataset(train_pth: Path, test_pth: Path, out_pth=None, overwrite=False):
    def split(_df: pd.DataFrame):
        cols = [o for o in re.split('[;"]', _df.columns.item()) if o != '']
        data = [[] for _ in range(len(cols))]
        for t in _df.values.flatten():
            t = [o for o in re.split('[;"]', t) if o != '']
            for i, it in enumerate(t):
                data[i].append(it)
        return pd.DataFrame(dict(zip(cols, data)))

    train_df = split(pd.read_csv(train_pth))
    test_df = split(pd.read_csv(test_pth))
    df = pd.concat([train_df, test_df])
    df.reset_index(drop=True, inplace=True)
    ids = ['N%05d' % i for i in range(len(df))]
    df.insert(0, 'ID', ids)
    df.set_index('ID', inplace=True)
    if overwrite:
        train_df.insert(0, 'ID', ids[:len(train_df)])
        test_df.insert(0, 'ID', ids[len(train_df):])
        train_df.set_index('ID', inplace=True)
        test_df.set_index('ID', inplace=True)
        train_df.to_csv(train_pth)
        test_df.to_csv(test_pth)
    if out_pth is not None:
        df.to_csv(out_pth)
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


if __name__ == '__main__':
    dir_pth = Path('bank')
    # df = preprocess_bank_dataset(dir_pth / 'train.csv', dir_pth / 'test.csv', dir_pth / 'full_bank.csv',
    #                              overwrite=True)
    df = pd.read_csv(dir_pth / 'full_bank.csv')
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    dicts = categorical_values_to_int(df, columns=cat_cols,
                                      given_dicts=[{'no': 0, 'yes': 1}])[1]
    df['pdays'] = df['pdays'].apply(lambda x: 999 if x == -1 else x)
    col_mappings = dict(zip(cat_cols, dicts))
    for col in ['age', 'balance', 'campaign', 'previous', 'pdays']:
        arr = df[col].values
        min_arr, max_arr = arr.min(), arr.max()
        df[col] = (arr - arr.min()) / (arr.max() - arr.min())
        col_mappings[col] = (arr.min(), arr.max())
    with open(dir_pth / 'mappings.pth', 'wb') as f:
        pickle.dump(col_mappings, f)
    cols = [['ID', 'housing', 'loan', 'contact', 'day', 'month',
             'campaign', 'pdays', 'previous', 'poutcome', 'y'],
            ['ID', 'default', 'balance'],
            ['ID', 'age', 'job', 'marital', 'education']]
    partition(df, dir_pth, cols, [2, 2])
