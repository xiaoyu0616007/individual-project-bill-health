# -*- coding: utf-8 -*-
"""
账单解析模块：微信/支付宝官方导出 CSV（或同结构的样例数据）→ 标准 DataFrame。
app.py 与 scripts/anonymize.py 共用本模块，保证线上应用与本地脱敏脚本走同一套解析逻辑。
"""
import pandas as pd


def _find_header_row(lines, max_scan=30):
    """微信/支付宝导出的 CSV 前几行是说明文字，扫描出真正表头所在行。"""
    keys = ("交易时间", "支付时间", "日期", "时间")
    for i, line in enumerate(lines[:max_scan]):
        if any(k in line for k in keys) and ("金额" in line or "金额(元)" in line):
            return i
    return 0


def parse_bill_bytes(raw: bytes) -> pd.DataFrame:
    """兼容三种格式：微信官方导出、支付宝官方导出、开发样例（对齐样例字段）。"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    return parse_bill_text(text)


def parse_bill_text(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = _find_header_row(lines)
    df = pd.read_csv(pd.io.common.StringIO("\n".join(lines[header_idx:])))

    # 字段名模糊匹配
    colmap = {}
    for c in df.columns:
        cs = str(c).strip()
        if cs in ("交易时间", "支付时间", "付款时间", "交易创建时间", "日期"):
            colmap["交易时间"] = c
        elif cs in ("金额", "金额(元)", "金额（元）", "交易金额"):
            colmap["金额"] = c
        elif cs in ("收/支", "收支", "资金流向", "收/付款"):
            colmap["收/支"] = c
        elif cs in ("交易分类", "分类", "交易类型", "类型"):
            colmap["交易分类"] = c
        elif cs in ("商品", "商品名称", "交易对方", "商品/商户"):
            colmap["商品"] = c
        elif cs in ("支付方式", "收/付款方式", "付款方式"):
            colmap["支付方式"] = c
    missing = [k for k in ("交易时间", "金额") if k not in colmap]
    if missing:
        raise ValueError(f"未识别到必需字段：{'、'.join(missing)}。请确认使用的是官方导出的账单文件。")

    out = pd.DataFrame()
    out["交易时间"] = pd.to_datetime(df[colmap["交易时间"]], errors="coerce")
    out["金额"] = (
        df[colmap["金额"]]
        .astype(str)
        .str.replace(r"[¥,\s]", "", regex=True)
        .astype(float)
    )
    out["交易分类"] = df[colmap["交易分类"]].fillna("其他").astype(str) if "交易分类" in colmap else "其他"
    out["商品"] = df[colmap["商品"]].astype(str) if "商品" in colmap else ""
    out["支付方式"] = df[colmap["支付方式"]].astype(str) if "支付方式" in colmap else ""

    if "收/支" in colmap:
        direction = df[colmap["收/支"]].astype(str)
        out = out[direction.str.contains("支", na=False) & ~direction.str.contains("不计", na=False)]
    out = out.dropna(subset=["交易时间"])
    out["金额"] = out["金额"].abs()
    out["月份"] = out["交易时间"].dt.to_period("M").astype(str)
    return out.reset_index(drop=True)


def parse_bill(uploaded) -> pd.DataFrame:
    """Streamlit UploadedFile（或任何带 getvalue() 的对象）。"""
    return parse_bill_bytes(uploaded.getvalue())
