# individual-project-bill-health

**家庭日常开销体检：基于账单可视化的个人与家庭理财辅助工具**

《数据分析与可视化》课程个人项目 · 西南财经大学 · 周笑宇（42519045）

上传微信/支付宝官方导出的账单，自动完成分类汇总、可视化呈现与环比异常检测，并由大语言模型生成大白话"开销体检报告"。让不擅长记账的个人及家庭用户，几分钟内获得过去需要记账数月才能形成的开销认知。

## 功能

- **账单解析**：兼容微信、支付宝官方全量导出的 CSV 格式，字段名模糊匹配
- **四张核心图表**：月度开销走势、分类开销 TOP6、开销结构环图、环比激增项检测（Plotly 交互图）
- **AI 体检报告**：大语言模型只基于程序算出的统计量解读（事实锚定），无 API key 时自动回退为纯统计模板报告
- **隐私设计**：原始账单文件不落盘；仅日期/分类/金额三个匿名字段存本地 SQLite，支持一键清除；外发内容可预览
- **建模基准**：基于 UCI 公开数据集（30,000 条）的逻辑回归 5 折交叉验证，AUC 指标

## 数据

| 数据 | 说明 |
|---|---|
| `data/sample_bill.csv` | 虚构样例账单（201 条，2026-05 至 2026-07），字段对齐官方导出格式，仅用于开发与演示 |
| `data/uci_credit_card.csv` | UCI Default of Credit Card Clients（30,000 条），用于分析与建模，来源：https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |

## 本地运行

```bash
pip install -r requirements.txt
# 可选：设置 DeepSeek API key 启用 AI 报告（不设置则使用纯统计模板报告）
set DEEPSEEK_API_KEY=你的key
streamlit run app.py
```

## 部署

代码推送到 GitHub 后，在 [Streamlit Community Cloud](https://share.streamlit.io) 选择本仓库即可部署；AI 功能的 key 配置在 Streamlit Secrets 中（不进代码库）。

## 隐私声明

- 不涉及任何资金交易建议，不构成投资或消费承诺
- 原始账单文件不落盘、不上传云端
- 本地仅保存日期、分类、金额三个匿名字段，可一键清除
- 发送给大语言模型的仅为聚合统计量，不含商户名与商品名
