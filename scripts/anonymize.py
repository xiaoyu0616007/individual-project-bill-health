# -*- coding: utf-8 -*-
"""
脱敏收集工具（命令行，零依赖 pandas 以外的第三方库）

用途：把微信/支付宝官方导出的账单 CSV，转换为只含三个匿名字段的 CSV：
    交易日期、交易分类、金额
商户名、商品名、支付方式、收/支等所有其他字段一律丢弃。输出文件不含任何
个人身份信息，可直接用于本项目的分析流程，也便于在不暴露隐私的前提下
收集多人的真实账单数据（多人各自脱敏后汇总，每人几千条、十人即上万条）。

用法（在仓库根目录下）：
    python scripts/anonymize.py 微信账单.csv
    python scripts/anonymize.py 微信账单.csv -o 我的匿名账单.csv
    python scripts/anonymize.py a.csv b.csv          # 批量处理
    缺省输出为原文件名后加 "_匿名"（与原文件同目录）。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bill_parser import parse_bill_bytes  # noqa: E402


def anonymize(src: Path, dst: Path) -> int:
    df = parse_bill_bytes(src.read_bytes())
    out = pd.DataFrame({
        "交易日期": df["交易时间"].dt.strftime("%Y-%m-%d"),
        "交易分类": df["交易分类"],
        "金额": df["金额"].round(2),
    })
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    return len(out)


def main():
    ap = argparse.ArgumentParser(
        description="微信/支付宝账单脱敏：仅保留 交易日期/交易分类/金额 三个匿名数字段"
    )
    ap.add_argument("inputs", nargs="+", help="微信或支付宝官方导出的 CSV 文件（可传多个）")
    ap.add_argument("-o", "--output", default=None, help="输出路径（仅单文件时有效）")
    args = ap.parse_args()

    for raw in args.inputs:
        src = Path(raw)
        if not src.exists():
            print(f"跳过（文件不存在）：{src}")
            continue
        if args.output and len(args.inputs) == 1:
            dst = Path(args.output)
        else:
            dst = src.with_name(src.stem + "_匿名.csv")
        try:
            n = anonymize(src, dst)
        except Exception as e:
            print(f"失败：{src.name} → {e}")
            continue
        print(f"完成：{src.name} → {dst.name}，共 {n} 条支出（仅日期/分类/金额，其余字段已丢弃）")


if __name__ == "__main__":
    main()
