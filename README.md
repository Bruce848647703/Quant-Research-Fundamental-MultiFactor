# Quant-Research-Fundamental-MultiFactor

A股量化研究项目 - **基本面多因子月度轮动策略**

在沪深300成分股内，用基本面五因子（价值 EP/BP、质量 ROE、小市值 Size、动量 Momentum）等权合成打分，每月末调仓，持有得分最高的30只股票，与沪深300基准对比。

## 策略逻辑

1. **股票池**: 沪深300成分股（约300只）
2. **因子构建**:
   | 因子 | 定义 | 方向 |
   |------|------|------|
   | EP | 1 / PE(TTM)，亏损股剔除 | 越大越好 |
   | BP | 1 / PB，破净为负剔除 | 越大越好 |
   | ROE | ROE(TTM)，累计加权ROE还原，按披露滞后对齐时点 | 越大越好 |
   | Size | -ln(总市值) | 市值越小越好 |
   | Momentum | 12-1月动量（252日收益剔除最近21日） | 越大越好 |
3. **因子预处理**: 每月末截面 MAD 去极值 + zscore 标准化
4. **合成**: 五因子等权加权（权重可在 `config/config.yaml` 调整）
5. **调仓**: 每月最后一个交易日，买入得分最高的30只，等权持有
6. **成本**: 佣金万三 + 印花税千一(卖出) + 滑点千一，按换手率计提

**ROE时点对齐**: 使用 `本期累计ROE + 上年年报ROE - 上年同期累计ROE` 还原TTM，并按财报披露法定期限（一季报+30天、半年报+62天、三季报+31天、年报+121天）滞后生效，避免未来函数。

## 数据来源

- 估值数据（PE/PB/总市值等日频，2018年起）: akshare `stock_value_em`（东方财富估值分析）
- 财务指标（季度加权ROE）: akshare `stock_financial_analysis_indicator_em`（东方财富财务分析）
- 基准指数: akshare `stock_zh_index_daily`（沪深300）
- 收益率由每日官方涨跌幅连乘构造，自动兼容除权除息

## 回测结果（2020-01 ~ 2026-07）

### 多头组合 vs 基准

| 指标 | 多头组合 (Top 30) | 沪深300 |
|------|------|------|
| 累计收益 | **+206.72%** | +14.59% |
| 年化收益 | **18.56%** | 2.09% |
| 年化波动 | 15.36% | - |
| 夏普比率 | 1.21 | - |
| 最大回撤 | -10.83% | -39.92% |
| 卡玛比率 | 1.71 | - |
| 平均月换手 | 12.91% | - |

![净值曲线](results/plots/nav_curve.png)

### 分层回测（5分组，第1组得分最高）

| 分组 | 年化收益 | 夏普 | 最大回撤 |
|------|------|------|------|
| group_1 (top) | 23.33% | 1.50 | -11.43% |
| group_2 | 23.13% | 1.16 | -16.84% |
| group_3 | 20.08% | 1.07 | -17.72% |
| group_4 | 15.73% | 0.73 | -20.89% |
| group_5 (bottom) | 10.47% | 0.45 | -25.86% |

分组收益自上而下单调递减，因子区分度良好。

![分层净值](results/plots/quantile_nav.png)

### 因子 Rank IC

| 因子 | IC均值 | ICIR | IC胜率 |
|------|------|------|------|
| size | +0.0540 | +0.301 | 60.3% |
| bp | +0.0292 | +0.102 | 56.4% |
| ep | +0.0234 | +0.083 | 47.4% |
| momentum | +0.0123 | +0.050 | 53.8% |
| roe | -0.0020 | -0.014 | 52.6% |

![累计IC](results/plots/factor_ic_cumsum.png)

## 项目结构

```
Quant-Research-Fundamental-MultiFactor/
├── config/config.yaml          # 全局配置（股票池/因子权重/回测参数）
├── data/
│   ├── raw/                    # 原始数据（parquet，已gitignore）
│   └── processed/              # 处理后数据
├── notebooks/                  # 研究notebook
├── results/
│   ├── reports/                # 回测报告与指标汇总
│   └── plots/                  # 图表
├── scripts/
│   ├── fetch_data.py           # 数据下载
│   └── run_backtest.py         # 一键回测
├── src/
│   ├── data/data_loader.py     # 数据获取与预处理
│   ├── factors/base.py         # 去极值/标准化
│   ├── factors/fundamental.py  # EP/BP/ROE/Size/Momentum 五因子
│   ├── backtest/engine.py      # 月度轮动回测引擎
│   ├── backtest/analyzers.py   # IC分析与报告生成
│   └── utils/helpers.py        # 工具函数
├── tests/test_factors.py       # 因子单元测试
├── requirements.txt
└── setup.py
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 1. 下载数据（沪深300估值+财务+基准指数，约5分钟）
python scripts/fetch_data.py

# 2. 运行回测（产出报告、图表与指标）
python scripts/run_backtest.py

# 3. 运行单元测试
pytest tests/ -v
```

## 注意事项

- 成分股使用当前时点名单，存在幸存者偏差，历史回测结果偏乐观
- 财报披露滞后使用法定期限近似，未使用实际披露日期
- 净值为月末采样，期间回撤可能被低估
- 本项目仅为量化研究用途，不构成任何投资建议
