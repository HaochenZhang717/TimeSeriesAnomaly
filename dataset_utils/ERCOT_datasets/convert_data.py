import pandas as pd
import numpy as np


def fix_24h_time(s):
    """
    把 '01/01/2021 24:00' -> '01/02/2021 00:00'
    """
    if isinstance(s, str) and "24:00" in s:
        # 拆日期
        date_part = s.replace(" 24:00", "")
        dt = pd.to_datetime(date_part)
        dt = dt + pd.Timedelta(days=1)
        return dt
    else:
        return pd.to_datetime(s)



def get_label_dataset():

    # -------- 配置 --------
    input_file = "Native_Load_2021.xlsx"
    output_file = "Native_Load_2021_labeled.xlsx"
    date_col = "Hour Ending"   # <<< 改成你的日期列名
    # ---------------------

    # 读 Excel
    df = pd.read_excel(input_file)

    # 确保日期列是 datetime


    df[date_col] = df[date_col].apply(fix_24h_time)


    # 定义区间（不限制年份，默认同一年）
    start_date = pd.to_datetime("02-12", format="%m-%d")
    end_date   = pd.to_datetime("02-19", format="%m-%d")

    # 取月-日进行比较
    md = df[date_col].dt.strftime("%m-%d")

    df["label"] = ((md >= "02-12") & (md <= "02-19")).astype(int)

    # 保存
    df.to_excel(output_file, index=False)

    print("✅ 完成！已生成:", output_file)
    print(df[[date_col, "label"]].head())


def get_npy_files():
    input_file = "Native_Load_2021_labeled.xlsx"
    # 读 Excel
    df = pd.read_excel(input_file)
    coast = np.array(df['COAST'])
    east = np.array(df['EAST'])
    fwest = np.array(df['FWEST'])
    north = np.array(df['NORTH'])
    ncent = np.array(df['NCENT'])
    south = np.array(df['SOUTH'])
    scent = np.array(df['SCENT'])
    west = np.array(df['WEST'])

    np.save("raw_data/coast.npy", coast)
    np.save("raw_data/east.npy", east)
    np.save("raw_data/fwest.npy", fwest)
    np.save("raw_data/north.npy", north)
    np.save("raw_data/ncent.npy", ncent)
    np.save("raw_data/south.npy", south)
    np.save("raw_data/scent.npy", scent)
    np.save("raw_data/west.npy", west)

    np.save("raw_data/label.npy", np.array(df['label']))
