import pandas as pd


def get_device_name(df_dev_map, devID):
    """dev_id에 해당하는 dev_nm(장치명)을 매핑 DataFrame에서 조회"""
    row = df_dev_map[df_dev_map['dev_id'] == devID]
    if row.empty:
        raise ValueError(f"devID {devID}에 해당하는 장치를 찾을 수 없습니다.")
    return row.iloc[0]['dev_nm']


def select_devID_tagCD(df, devID, tagCD):
    return df[(df['dev_id'] == devID) & (df['tag_cd'] == tagCD)]


def print_tagCD(df, devID):
    df_devID = df[df['dev_id'] == devID]
    print('tagCD list: ', df_devID['tag_cd'].unique())
