# -*- coding: utf-8 -*-
"""
家庭日常开销体检：基于账单可视化的个人与家庭理财辅助工具
流程：上传账单（微信/支付宝官方导出或样例） -> 解析 -> 分类汇总 -> 环比异常检测
      -> 4 张 Plotly 图表 -> 大语言模型体检报告（无 key 时回退纯统计模板）
隐私：原始账单文件不落盘；仅日期/分类/金额三个匿名字段存本地 SQLite，可一键清除。
"""
import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="家庭日常开销体检", page_icon="🩺", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
FALLBACK_REPORT = None  # 无 AI key 时的纯统计模板报告在 build_fallback() 中生成


# ---------------------------------------------------------------- 解析
def _find_header_row(lines, max_scan=30):
    """微信/支付宝导出的 CSV 前几行是说明文字，扫描出真正表头所在行。"""
    keys = ("交易时间", "支付时间", "日期", "时间")
    for i, line in enumerate(lines[:max_scan]):
        if any(k in line for k in keys) and ("金额" in line or "金额(元)" in line):
            return i
    return 0


def parse_bill(uploaded) -> pd.DataFrame:
    """兼容三种格式：微信官方导出、支付宝官方导出、开发样例（对齐样例字段）。"""
    raw = uploaded.getvalue()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8-sig", errors="replace")

    lines = text.splitlines()
    header_idx = _find_header_row(lines)
    df = pd.read_csv(pd.io.common.StringIO("\n".join(lines[header_idx:])))

    # 字段名模糊匹配（对应风险五的应对动作）
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


# ---------------------------------------------------------------- 统计
def summarize(df: pd.DataFrame):
    monthly = df.groupby("月份")["金额"].sum()
    cat_month = df.pivot_table(index="交易分类", columns="月份", values="金额", aggfunc="sum").fillna(0)
    if len(cat_month.columns) >= 2:
        last, prev = cat_month.columns[-1], cat_month.columns[-2]
        mom = (cat_month[last] - cat_month[prev]).sort_values(ascending=False)
    else:
        last, prev, mom = cat_month.columns[-1], None, pd.Series(dtype=float)
    return monthly, cat_month, mom, last, prev


# ---------------------------------------------------------------- 匿名存储（三字段、可一键清除）
def get_db_path() -> str:
    if "db_path" not in st.session_state:
        st.session_state.db_path = os.path.join(tempfile.gettempdir(), f"bill_health_{os.getpid()}.db")
    return st.session_state.db_path


def save_anonymous(df: pd.DataFrame):
    conn = sqlite3.connect(get_db_path())
    conn.execute(
        "CREATE TABLE IF NOT EXISTS anon (d TEXT, cat TEXT, amt REAL)"
    )
    conn.execute("DELETE FROM anon")
    conn.executemany(
        "INSERT INTO anon VALUES (?,?,?)",
        zip(df["交易时间"].dt.strftime("%Y-%m-%d"), df["交易分类"], df["金额"]),
    )
    conn.commit()
    conn.close()


def clear_db():
    path = get_db_path()
    if os.path.exists(path):
        os.remove(path)
    st.session_state.pop("db_path", None)


# ---------------------------------------------------------------- 图表
def make_charts(monthly, cat_month, mom, last, prev):
    figs = []
    figs.append(px.line(monthly.reset_index(), x="月份", y="金额", markers=True,
                        title="月度开销走势（元）"))
    top = cat_month[last].sort_values(ascending=False).head(6)
    figs.append(px.bar(top.reset_index(), x=last, y="交易分类", orientation="h",
                       title=f"{last} 分类开销 TOP6（元）").update_yaxes(autorange="reversed"))
    figs.append(px.pie(names=top.index, values=top.values, title=f"{last} 开销结构"))
    if prev is not None and len(mom[mom > 0]) > 0:
        surge = mom[mom > 0].head(3).rename("环比增加（元）").reset_index()
        figs.append(px.bar(surge, x="环比增加（元）", y="交易分类", orientation="h",
                           title=f"环比激增项 TOP3（{last} vs {prev}）").update_yaxes(autorange="reversed"))
    else:
        surge = None
    return figs, surge


# ---------------------------------------------------------------- AI 报告（事实锚定）
def build_stats(monthly, cat_month, mom):
    return {
        "近几个月总开销": monthly.round(2).to_dict(),
        "最近月分类明细": cat_month.iloc[:, -1].round(2).to_dict(),
        "环比变化": mom.round(2).to_dict(),
    }


def llm_report(stats) -> str | None:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        try:
            key = st.secrets.get("DEEPSEEK_API_KEY", None)
        except Exception:
            key = None
    if not key:
        return None
    try:
        from openai import OpenAI
        prompt = f"""
你是一位面向社区家庭的理财科普助手。下面是一位用户的家庭月度开销统计（金额单位：元，已脱敏）：
{stats}
请输出一份大白话"开销体检报告"：
1. 一句话总结最近的整体情况；
2. 指出花得最多的 2 个分类和环比涨幅最大的分类，用生活化语言解释可能的原因（只做假设，不下结论）；
3. 给出 1-3 条具体、可执行的省钱建议（如"每周少点 2 次外卖可省约 X 元"）；
4. 最后提醒：本报告仅基于账单数据，不构成投资或消费承诺。
只依据上面给出的数字回答，不要编造额外数据。用中文。
"""
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.warning(f"大语言模型调用失败，已回退为纯统计模板报告（{type(e).__name__}）")
        return None


def build_fallback(monthly, cat_month, mom, last, prev) -> str:
    top = cat_month[last].sort_values(ascending=False)
    top2 = "、".join(f"{c} {v:.0f} 元" for c, v in top.head(2).items())
    surge_txt = "暂无环比数据（仅上传了一个月的账单）"
    if prev is not None and len(mom[mom > 0]) > 0:
        s = mom[mom > 0].head(1)
        surge_txt = f"{s.index[0]}环比增加约 {s.iloc[0]:.0f} 元"
    total = monthly.sum()
    lines = [
        f"**一句话总结**：样本期内共支出 {total:.0f} 元，最近月（{last}）支出 {top.sum():.0f} 元。",
        f"**花得最多**：{top2}。",
        f"**环比变化**：{surge_txt}。",
        "**建议**：可先从支出最高的分类入手，查看是否有可削减的非必要项；对环比激增的分类核对是否为一次性支出。",
        "",
        "*本报告仅基于账单数据的统计解读，不构成投资或消费承诺。*",
    ]
    return "\n\n".join(lines)


# ---------------------------------------------------------------- 建模（第三方数据集）
@st.cache_data(show_spinner=False)
def train_baseline() -> float:
    """UCI Default of Credit Card Clients：逻辑回归 5 折交叉验证 AUC（指标七）。"""
    path = DATA_DIR / "uci_credit_card.csv"
    if not path.exists():
        return float("nan")
    df = pd.read_csv(path)
    y = (df["default payment next month"] == 1).astype(int)
    feats = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
             "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3"]
    X = df[feats].fillna(0)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


# ---------------------------------------------------------------- 页面
st.title("🩺 家庭日常开销体检")
st.caption("上传账单 → 自动分类汇总 → 4 张图表 → AI 体检报告。原始文件不落盘，匿名统计可一键清除。")

with st.expander("🔐 隐私承诺（请先读这三句）", expanded=True):
    st.markdown(
        "1. **原始账单文件不保存**：仅在当前页面会话内解析，不上传云端、不写入服务器磁盘；\n"
        "2. **只存三个匿名字段**：日期、分类、金额存入本地 SQLite，商户名与商品名从不写入，点下方按钮随时一键清除；\n"
        "3. **发给 AI 的只有汇总数字**：如“餐饮外卖 1400 元”，可勾选“发送内容预览”查看全部外发内容。"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ 一键清除本地匿名数据", use_container_width=True):
            clear_db()
            st.success("本地匿名数据已清除。")
    with col2:
        preview_on = st.checkbox("发送内容预览", value=False, help="勾选后展示将发送给大语言模型的统计内容")

with st.expander("📥 不会导出账单？三步搞定（微信/支付宝）"):
    st.markdown(
        "**微信**：我 → 服务 → 钱包 → 账单 → 右上角常见问题 → 下载账单 → 用作证明材料 → 申请\n\n"
        "**支付宝**：我的 → 账单 → 右上角 ··· → 开具交易流水证明 → 用作证明材料 → 申请\n\n"
        "两个平台都支持导出**全部历史记录**，导出的 CSV 文件直接拖进下面的上传框即可。"
    )

uploaded = st.file_uploader("上传账单 CSV 文件", type=["csv"])
use_sample = st.button(" or 先看样例数据的体检报告", type="secondary")

df = None
source = None
if uploaded is not None:
    try:
        df = parse_bill(uploaded)
        source = uploaded.name
    except Exception as e:
        st.error(f"解析失败：{e}")
elif use_sample or st.session_state.get("sample_loaded"):
    df = pd.read_csv(DATA_DIR / "sample_bill.csv", parse_dates=["交易时间"])
    df = df[df["收/支"] == "支出"].copy()
    df["月份"] = df["交易时间"].dt.to_period("M").astype(str)
    source = "sample_bill.csv（虚构样例）"
    st.session_state.sample_loaded = True

if df is not None and len(df) > 0:
    st.success(f"解析成功：**{len(df)}** 条支出记录，来源 {source}。数据范围 "
               f"{df['交易时间'].min().date()} 至 {df['交易时间'].max().date()}。")
    save_anonymous(df)

    monthly, cat_month, mom, last, prev = summarize(df)
    figs, surge = make_charts(monthly, cat_month, mom, last, prev)

    # ---- AI 报告卡片 ----
    stats = build_stats(monthly, cat_month, mom)
    if preview_on:
        with st.expander("将发送给大语言模型的全部内容", expanded=True):
            st.code(str(stats), language="python")
    with st.spinner("正在生成体检报告…"):
        report = llm_report(stats) or build_fallback(monthly, cat_month, mom, last, prev)
    st.subheader("📋 AI 开销体检报告")
    st.info(report)
    if st.session_state.get("ai_used") is None:
        st.session_state.ai_used = report is not None

    # ---- 4 张图 ----
    st.subheader("📊 四张核心图表")
    c1, c2 = st.columns(2)
    for i, fig in enumerate(figs):
        (c1 if i % 2 == 0 else c2).plotly_chart(fig, use_container_width=True)

    # ---- 建模基准 ----
    with st.expander("🔬 第三方数据集建模基准（UCI 30,000 条，逻辑回归 5 折交叉验证）"):
        auc = train_baseline()
        if auc == auc:
            st.metric("模型性能 AUC", f"{auc:.3f}", delta="目标 ≥ 0.70",
                      delta_color="normal" if auc >= 0.70 else "inverse")
            st.caption("数据来源：UCI Default of Credit Card Clients（Dua & Graff, 2019）。"
                       "该模型仅用于分析与建模方法验证，不用于你的账单。")
        else:
            st.warning("未找到 data/uci_credit_card.csv")
else:
    st.markdown(
        "点击上方 **“先看样例数据的体检报告”** 体验完整流程，或上传你自己的账单文件。\n\n"
        "样例为虚构数据（201 条，2026 年 5-7 月），字段结构对齐微信/支付宝官方导出格式。"
    )
