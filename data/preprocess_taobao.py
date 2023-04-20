import pandas as pd
from pathlib import Path
import pickle

from data.preprocessing_utils import vertical_split, horizontal_split, categorical_values_to_int


def load_taobao_df():
    ad = "taobao/ad_feature.csv"
    ad_feature_df = pd.read_csv(ad)

    raw_sample = "taobao/raw_sample.csv"
    raw_sample_df = pd.read_csv(raw_sample)

    user = "taobao/user_profile.csv"
    user_profile_df = pd.read_csv(user)

    # memory optimize for ad feature dataframe
    optimized_gl = raw_sample_df.copy()

    gl_int = raw_sample_df.select_dtypes(include=['int'])
    converted_int = gl_int.apply(pd.to_numeric, downcast='unsigned')
    optimized_gl[converted_int.columns] = converted_int

    gl_obj = raw_sample_df.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()
    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:, col] = gl_obj[col].astype('category')
        else:
            converted_obj.loc[:, col] = gl_obj[col]
    optimized_gl[converted_obj.columns] = converted_obj
    raw_sample_df = optimized_gl.copy()
    raw_sample_df_new = raw_sample_df.rename(columns={"user": "userid"})
    optimized_g2 = ad_feature_df.copy()
    g2_int = ad_feature_df.select_dtypes(include=['int'])
    converted_int = g2_int.apply(pd.to_numeric, downcast='unsigned')
    optimized_g2[converted_int.columns] = converted_int

    g2_float = ad_feature_df.select_dtypes(include=['float'])
    converted_float = g2_float.apply(pd.to_numeric, downcast='float')
    optimized_g2[converted_float.columns] = converted_float

    optimized_g3 = user_profile_df.copy()

    g3_int = user_profile_df.select_dtypes(include=['int'])
    converted_int = g3_int.apply(pd.to_numeric, downcast='unsigned')
    optimized_g3[converted_int.columns] = converted_int

    g3_float = user_profile_df.select_dtypes(include=['float'])
    converted_float = g3_float.apply(pd.to_numeric, downcast='float')
    optimized_g3[converted_float.columns] = converted_float

    # combine 3 tables
    df1 = raw_sample_df_new.merge(optimized_g3, on="userid")
    final_df = df1.merge(optimized_g2, on="adgroup_id")

    final_df['pvalue_level'] = final_df['pvalue_level'].fillna(2, )
    final_df['final_gender_code'] = final_df['final_gender_code'].fillna(1, )
    final_df['age_level'] = final_df['age_level'].fillna(3, )
    final_df['shopping_level'] = final_df['shopping_level'].fillna(2, )
    final_df['occupation'] = final_df['occupation'].fillna(0, )
    final_df['brand'] = final_df['brand'].fillna(0, )
    final_df['customer'] = final_df['customer'].fillna(0, )
    final_df['cms_group_id'] = final_df['cms_group_id'].fillna(13, )

    final_df['pvalue_level'] -= 1
    final_df['shopping_level'] -= 1
    final_df = final_df.astype({"cms_segid": int,
                                "cms_group_id": int,
                                'clk': int,
                                'adgroup_id': int,
                                'final_gender_code': int,
                                'age_level': int,
                                'pvalue_level': int,
                                'shopping_level': int,
                                'occupation': int,
                                'cate_id': int,
                                'customer': int,
                                'brand': int}
                               )

    return final_df


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
    # import os
    # print(os.getcwd())
    # df = load_taobao_df()
    # df.to_csv('taobao/preprocessed.csv')
    # pick 50k samples
    # df = pd.read_csv('taobao/preprocessed.csv')
    # df = df.head(50000)
    # df.to_csv('taobao/head_50k.csv', index=False)

    # df = pd.read_csv('taobao/head_50k.csv')
    # df = df.drop(columns=['Unnamed: 0'])
    # df.reset_index(drop=True, inplace=True)
    # ids = ['N%05d' % i for i in range(len(df))]
    # df.insert(0, 'ID', ids)
    # df.set_index('ID', inplace=True)
    # df.to_csv('taobao/head_50k.csv', index=True)

    df = pd.read_csv('taobao/head_50k.csv')
    data_dir = Path('taobao')
    df['new_user_class_level '] = df['new_user_class_level '].fillna(0, )
    cat_cols = [
        'pid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level',
        'shopping_level', 'occupation', 'new_user_class_level ', 'cate_id', 'brand'
    ]
    num_cols = ['price']
    label_col = 'clk'
    df = df.loc[:, ['ID'] + cat_cols + num_cols + [label_col]]
    print(df.columns)
    for col in cat_cols:
        lst = sorted(df[col].unique().tolist())
        mapping = dict(zip(lst, range(len(lst))))
        df[col] = df[col].apply(lambda x: mapping[x])
    for col in num_cols:
        arr = df[col].values
        min_arr, max_arr = arr.min(), arr.max()
        df[col] = (arr - arr.min()) / (arr.max() - arr.min())
    cols = [
        ['ID', 'pid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level',
         'shopping_level', 'occupation', 'new_user_class_level ', 'cate_id', 'brand', 'price', 'clk'],
        ['ID', 'final_gender_code', 'age_level', 'occupation'],
        ['ID', 'pvalue_level', 'shopping_level']
    ]
    partition(df, data_dir, cols, [2, 2])
