# -*- coding: utf-8 -*-
"""
数据转换脚本：UCI Online Retail → data/online_retail_daily.csv

原始数据集（约 22 MB 压缩包）从 UCI 官方通道免费下载：
    https://archive.ics.uci.edu/dataset/352/online+retail
转换逻辑：
    1. 过滤数量与单价为正的有效交易（原始 541,909 条 → 有效 530,104 条）；
    2. 商品描述关键词近似归入 8 个类别（餐厨茶具、家居装饰、玩具游戏、文具卡片、
       包袋配饰、收纳储物、布艺纺织、其他）；
    3. 按"交易日期 × 类别"聚合为 2,440 组，输出字段与脱敏收集工具的输出同构：
       交易日期、交易分类、金额。

用法（在仓库根目录）：
    python scripts/make_retail_slim.py C:\\下载\\online+retail.zip
"""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

BUCKETS = [
    ("餐厨茶具", r"MUG|CUP|PLATE|BOWL|KITCHEN|TEA|COFFEE|BOTTLE|JAR|CUTLERY|LUNCH|DISH|TRAY|PITCHER|SALT|PEPPER"),
    ("家居装饰", r"LIGHT|LAMP|CANDLE|FRAME|PICTURE|ORNAMENT|VASE|CLOCK|MIRROR|DECORATION|WALL|HOOK|GARLAND|T-LIGHT|DOORSTOP|SCONCE|BIRDHOUSE"),
    ("玩具游戏", r"TOY|GAME|PUZZLE|DOLL|PLAYHOUSE|SKITTLE|DART|PLAYING|PLAYGROUND|BLOCK|TEDDY|PLUSH"),
    ("文具卡片", r"PEN|PENCIL|NOTEBOOK|STATIONERY|CARD|ENVELOPE|RULER|ERASER|STAMP|RIBBON|STICKER|BOOK|GIFT TAG|WRAP|SHARPIE"),
    ("包袋配饰", r"BAG|PURSE|WALLET|UMBRELLA|KEY|CHARM|BRACELET|NECKLACE|HAIRBAND|JEWEL|TRINKET|PENCIL CASE"),
    ("收纳储物", r"BOX|TIN|BASKET|STORAGE|DRAWER|ORGANIS|TRUNK|CRATE|SUITCASE|HANGING"),
    ("布艺纺织", r"APRON|TOWEL|CUSHION|BLANKET|RUG|PILLOW|CURTAIN|BUNTING|CLOTH|FABRIC|KNIT"),
]


def bucket(desc) -> str:
    d = str(desc).upper()
    for name, pat in BUCKETS:
        if re.search(pat, d):
            return name
    return "其他"


def main(zip_path: str):
    zip_file = Path(zip_path)
    if not zip_file.exists():
        raise SystemExit(f"未找到压缩包：{zip_path}（请先从 UCI 官方通道下载）")
    out = Path(__file__).resolve().parent.parent / "data" / "online_retail_daily.csv"

    with zipfile.ZipFile(zip_file) as z:
        xlsx = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = z.extract(xlsx, path=td)
            df = pd.read_excel(target, engine="openpyxl")

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["交易日期"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    df["金额"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["交易分类"] = df["Description"].map(bucket)

    daily = df.groupby(["交易日期", "交易分类"], as_index=False)["金额"].sum()
    daily = daily.sort_values(["交易日期", "交易分类"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"完成：{len(df):,} 条有效交易 → {out.name}（{len(daily):,} 组，"
          f"{df['交易日期'].min():%Y-%m-%d} 至 {df['交易日期'].max():%Y-%m-%d}）")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
