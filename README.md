# 🩺 家庭日常开销体检

**基于账单可视化的个人与家庭理财辅助工具**

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-交互图表-3F3F8F)](https://plotly.com/python/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-逻辑回归-F7931E)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![在线演示](https://img.shields.io/badge/在线演示-立即体验-brightgreen)](https://individual-project-bill-health-8bx2jkyfzw3kzfgr5hoyah.streamlit.app/)

《数据分析与可视化》课程个人项目 · 西南财经大学 · 周笑宇

---

## 目录

- [这是什么](#这是什么)
- [功能一览](#功能一览)
- [数据与方法](#数据与方法)
- [隐私设计](#隐私设计)
- [快速开始](#快速开始)
- [AI 能力说明](#ai-能力说明)
- [目录结构](#目录结构)
- [云端部署](#云端部署)
- [免责声明与许可](#免责声明与许可)

---

## 这是什么

很多家庭的日常开支由一名成员集中管理，通常是退休或临近退休的阿姨。她们每天经手买菜、日用品、水电燃气、话费宽带等数十笔小额支出，到月末汇总时往往只能留下一个大致印象，难以说清各项开销的具体去向。

通过社区调研，我确认了三个层层递进的痛点：

1. **手动记账坚持不下来**：笔数太多、时间成本太高，记几个月就会放弃；
2. **数据在沉淀，理解没有发生**：微信与支付宝已经自动记录了每一笔交易，但账单页只是流水罗列，用户很难得出"这个月外卖多花了 200 元"这样的结论；
3. **知道了也未必知道怎么办**：从统计结论到"每周少点两次外卖"之间，还差一层生活化的翻译。

本项目提供一个**免费的网页工具**：上传账单文件即得"体检报告"，自动完成分类汇总、可视化呈现与环比异常检测，并由大语言模型生成大白话解读与可执行建议。面向所有有开销记录与解读需求的个人及家庭用户，包括管理家庭开支的成员，也包括想理清个人账单的年轻人。

## 功能一览

| 模块 | 能做什么 | 实现 |
|---|---|---|
| 📥 账单解析 | 兼容微信、支付宝官方导出的 CSV，字段名模糊匹配，报错时给出明确提示 | pandas |
| 📊 四张核心图表 | 月度开销走势、分类开销 TOP6、开销结构环图、环比激增项检测（自动找出环比增加最大的 3 类） | Plotly 交互图 |
| 📋 AI 体检报告 | 一句话总结 + 花费最多分类 + 环比激增项 + 1 至 3 条可执行建议，四段式结构 | 大语言模型（DeepSeek） |
| 🔒 隐私三承诺 | 原始文件不落盘、匿名字段一键清除、外发内容可预览 | 会话内存 + 本地 SQLite |
| 📈 建模基准 | 在 UCI 公开数据集（30,000 条）上以 5 折交叉验证训练逻辑回归，展示 AUC | scikit-learn |
| 🧭 导出手册 | 页内提供微信/支付宝三步导出指引，不熟悉操作的用户也能完成 | Streamlit 组件 |

设计上有一条贯穿的**事实锚定**原则：大语言模型只接收程序预先算出的统计量（月度总额、分类明细、环比变化），不接触原始账单明细，并在提示词中约束"只依据给出的数字回答"，使报告中的每个数字都有据可查。

## 数据与方法

**数据来源分三类，全部免费可得：**

| 数据 | 来源 | 用途 |
|---|---|---|
| 用户账单 | 微信「我→服务→钱包→账单→常见问题→下载账单」、支付宝「我的→账单→开具交易流水证明」，均为官方全量导出通道 | 实际分析对象 |
| 开发样例 `data/sample_bill.csv` | 自行构造的虚构数据（201 条，2026 年 5 至 7 月），字段结构对齐官方导出格式 | 开发与演示 |
| 第三方数据集 `data/uci_credit_card.csv` | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)（Dua & Graff, 2019），30,000 条个人信用卡账单与还款记录 | 分析与建模 |

### UCI 数据集说明

UCI 机器学习仓库（UCI Machine Learning Repository）是加州大学欧文分校运营的公开学术数据集库，1987 年创立，是机器学习领域使用最广泛的公共数据来源之一。本项目使用的《Default of Credit Card Clients》是其中的经典数据集，原始数据来自台湾某银行 2005 年 4 月至 9 月信用卡客户的真实脱敏记录，配套有正式发表的学术论文（Yeh & Lien, 2009），可通过上方链接随时打开官网核对。

**数据概况：**

| 项目 | 数值 |
|---|---|
| 规模 | 30,000 行 × 25 列，每位客户一行，无缺失值 |
| 字段内容 | 额度、性别、年龄、婚否、学历，以及近 6 个月的账单金额（BILL_AMT1-6）、还款金额（PAY_AMT1-6）、还款状态（PAY_0 至 PAY_6） |
| 目标变量 | `default payment next month`（下月是否违约），违约率 22.1% |
| 年龄分布 | 21 至 79 岁，平均 35.5 岁 |

该数据集在本项目中承担分析与建模部分：取其中 14 个特征训练逻辑回归预测违约风险，5 折交叉验证评估，实测 AUC = 0.721。数据本身覆盖的是信用卡账单场景，与家庭日常开销有所不同，因此建模用于呈现一套完整的分析与建模流程，用户账单的分析则基于描述统计与环比检测。

引用格式：Dua, D., & Graff, C. (2019). UCI Machine Learning Repository: Default of Credit Card Clients Data Set. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

**分析方法：**

- **描述统计**：按月与按分类汇总，环比检测取"最近月减上月"差值排序，自动标记激增项；
- **建模**：以账单额度、账单金额、还款金额、还款状态等 14 个特征，用逻辑回归预测违约风险，5 折交叉验证评估，实测 AUC = 0.721；选择逻辑回归而非深度学习，原因是表格型数据已经够用、系数可直接解读、训练成本低；
- **可视化原则**：分类比较用条形图（位置编码比角度更易判断），结构占比只在少数类别时用环图，参考 Cleveland & McGill (1984) 与 Few (2013)。

**存储方案（分级）：** 第三方数据集与样例 CSV 随仓库落盘；用户原始账单文件不落盘；解析结果仅将日期、分类、金额三个匿名字段写入本地 SQLite，可一键清除。

## 隐私设计

页面顶部常驻三句承诺：

1. **原始账单文件不保存**：仅在当前页面会话内解析，不写入服务器磁盘；
2. **只存三个匿名字段**：日期、分类、金额，商户名与商品名从不写入，随时一键清除；
3. **发给 AI 的只有汇总数字**：如"餐饮外卖 1400 元"，可勾选"发送内容预览"查看全部外发内容。

账单导出文件本身不含姓名、电话、住址等身份信息，从源头消除最常见的泄露担忧。

## 快速开始

**方式一：在线使用（推荐）**

直接访问 **https://individual-project-bill-health-8bx2jkyfzw3kzfgr5hoyah.streamlit.app/** ，打开即可看到样例数据生成的完整体检报告，无需注册、无需配置任何东西；上传自己的账单 CSV 后，图表与报告即时切换为真实结果。

**方式二：本地运行**

```bash
git clone https://github.com/xiaoyu0616007/individual-project-bill-health.git
cd individual-project-bill-health
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

打开终端提示的本地地址，点击「先看样例数据的体检报告」即可体验完整流程，无需任何配置。

**方式三：用自己的账单**

按页面上的三步导出手册从微信或支付宝导出 CSV，拖进上传框，图表与报告即时重算刷新。

## AI 能力说明

| 能力 | 未配置 Key | 配置 DeepSeek Key |
|---|---|---|
| 账单解析、四张图表、环比检测、建模基准 | ✅ 完整可用 | ✅ 完整可用 |
| AI 体检报告 | ✅ 自动回退为纯统计模板报告（基于同一套计算结果生成，不含大语言模型） | ✅ 大语言模型生成四段式解读 |
| 外发内容预览、一键清除 | ✅ | ✅ |

Key 的读取顺序：环境变量 `DEEPSEEK_API_KEY`（本地）→ Streamlit Secrets（云端）。**代码库中不内置任何密钥**，`.env` 已在 `.gitignore` 中排除。图表与统计功能不依赖 AI，大模型不可用时产品主干不受影响。

## 目录结构

```
individual-project-bill-health/
├── app.py                    # 应用本体：解析、汇总、图表、AI 报告、匿名存储、建模基准
├── requirements.txt          # 运行依赖（锁定版本）
├── README.md                 # 本文件
├── .gitignore                # 排除 .env / .venv/ / __pycache__/ 等
└── data/
    ├── sample_bill.csv       # 开发样例账单（虚构，201 条）
    └── uci_credit_card.csv   # UCI 第三方数据集（30,000 条，用于分析与建模）
```

## 云端部署

部署平台：Streamlit Community Cloud（免费），直连本仓库，`git push` 后自动更新。

- **仓库地址**：https://github.com/xiaoyu0616007/individual-project-bill-health
- **应用地址**：https://individual-project-bill-health-8bx2jkyfzw3kzfgr5hoyah.streamlit.app/
- **分支 / 入口**：`main` / `app.py`

更新应用后执行 `git add -A`、`git commit -m "更新应用"`、`git push origin main` 即可同步上线。

## 免责声明与许可

- 本项目为课程学习与研究用途的理财辅助工具，**不涉及任何资金交易建议，不构成投资或消费承诺**；
- 人工智能输出固定附带"仅基于账单数据的统计解读"提示，报告中的归因均为假设而非结论；
- 样例数据全部虚构并标注；第三方数据集来自 UCI 公开库，仅用于分析与建模方法验证；
- 开源采用 MIT 许可，如需引用请注明课程与作者信息。

**参考资料（节选）：**

- Cleveland, W. S., & McGill, R. (1984). Graphical perception. *JASA*, 79(387), 531–554.
- Few, S. (2013). *Information Dashboard Design* (2nd ed.). Analytics Press.
- Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository: Default of Credit Card Clients*.
- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
