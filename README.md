# A股月频截面 Alpha 研究

这个项目研究的不是“大盘什么时候涨”，而是一个更窄、也更容易被证伪的问题：在月末已知的信息中，能否稳定识别下个月相对同业表现更好的股票。

当前版本不使用大盘择时改善净值，也不把数据清洗或复杂模型本身当成 Alpha。研究顺序固定为：经济假设 → 时间对齐 → 透明线性基线 → 截面检验 → 非线性增量检验 → 组合与成本 → 容量与失败条件。

## 1. 可证伪的 Alpha 假设

模型只使用月末或月末以前可获得的信息，预测股票下一月在所属行业内的收益排名。因子分为四类：

- 估值补偿：盈利收益率、账面市值比、销售市值比；
- 盈利质量：ROE、ROA、毛利率、现金转化以及盈利改善；
- 风险与交易行为：低波动、下行风险、低换手、极端收益规避和非流动性溢价；
- 价格行为：1个月、3个月、6个月和12-1个月反转，以及波动调整后的残差反转。

所有原始特征先在月度行业截面内转换为分位数。特征的代码定义、预期经济方向以及形成期/样本外 IC 都保存在 `output/factor_direction_audit.csv`。方向不由全样本收益反向挑选；弱且不稳定的低杠杆特征已经退出模型。

## 2. 标签与时间边界

- 信号时点：月末；
- 预测目标：下一月股票收益在当月所属行业内的分位数减 0.5；
- 财务数据：公告日以后才允许进入特征；
- 训练：只使用预测月以前且已实现标签的样本；
- 验证：每次重估只用历史训练窗口末尾12个月；
- 测试：2018-01 至 2025-11 形成信号，对应 2018-02 至 2025-12 收益，共95个月；
- 重估：每3个月一次，训练窗口最长84个月。

输入面板由 `data_cleaning.py` 统一生成。数据清洗是回测成立的必要条件，不作为策略收益来源或项目核心结论。

## 3. 模型顺序：清洗、普通回归、再做机器学习实验

所有方法必须先经过同一条 `clean_monthly_panel → prepare_advanced` 数据管线；任何模型都不能直接读取未清洗原始字段。随后按固定顺序比较：普通多因子 OLS → Ridge → Huber 稳健回归 → TabM 神经网络 → 历史相似样本检索。五种方法使用相同因子、标签、股票池、84个月训练窗、末尾12个月历史验证窗和每3个月重估规则。

Ridge 仍是正式候选，因为它在低信噪比截面上提供稳定正则化；OLS 是最透明的普通多因子回归基线。Huber、TabM 和检索模型只作实验对照，不能依据2024—2025回顾期结果切换生产模型。逐月结果见 `output/model_comparison_monthly.csv`，汇总见 `output/model_comparison_metrics.csv`。

| 模型 | 角色 | 平均Rank IC | 年化ICIR | 行业中性多空毛Sharpe |
|---|---|---:|---:|---:|
| OLS | 普通多因子回归基线 | 0.1217 | 3.290 | 1.629 |
| Ridge | 正式候选 | 0.1210 | 3.240 | 1.577 |
| Huber | 稳健回归实验 | 0.1218 | 3.294 | 1.650 |
| TabM | 机器学习实验 | 0.0582 | 2.917 | 0.833 |
| 历史相似样本 | 检索实验 | 0.0426 | 2.565 | 0.501 |

OLS、Ridge、Huber差异很小，不能据此声称稳健回归形成新 Alpha；TabM 和检索模型明显弱于线性回归，没有提供稳定增量。因此本轮不增加新模型权重，也不把复杂度当成收益来源。

## 4. 两个不同用途的组合

### 可交易近似：A股多头组合

- 在全部合格股票中持有预测最高的10%；
- 等权配置；
- 现有持仓跌出前15%才卖出，以降低换手；
- 月度单边成本基准为20bp；
- 这是现货多头研究组合，不靠大盘仓位开关改善结果。

### Alpha 诊断：行业中性多空组合

- 仅使用至少10只合格股票的行业；
- 每个行业配置相同资本；
- 行业内做多预测最高10%、做空最低10%，两侧分别等权；
- 多空两侧按真实权重变化计算换手和成本。

这个组合用于隔离截面排序能力，不等于可以直接在A股现货执行。真实做空需要融券可得性、借券费、保证金、基差以及拥挤度模型；当前数据不包含这些约束，因此 README 只把它称为“诊断组合”。

## 5. 评价指标与淘汰条件

项目不再只看多头组合夏普。核心证据包括：

- 月度截面 Rank IC、年化 ICIR 和正 IC 月份比例；
- 十分组收益的单调性；
- 行业中性多空收益、t统计量和分时期稳定性；
- Ridge 与最终组合的 IC 差值，用来判断复杂模型是否真正增加信息；
- 0/20/50/100/150/200bp 成本压力；
- 按5个交易日执行估算的 ADV 参与率。

下列任一情况都会阻止“可实盘”结论：多空 t 值低于2；主要子样本失效；分组收益不单调；非线性模型没有稳定验证增量；合理成本后夏普不足1；容量、融券或成交约束没有数据支持。

## 6. 完整样本结果

所有收益均为历史回测，不代表未来表现。多空诊断因最早两个月没有足够的行业组合收益，共覆盖93个月；其余序列覆盖95个月。

| 组合 | 成本 | 年化收益 | 年化波动 | Sharpe（rf=0） | 最大回撤 | 月均换手 |
|---|---:|---:|---:|---:|---:|---:|
| 多头组合 | 20bp | 10.69% | 18.92% | 0.565 | -25.38% | 34.86% |
| 行业中性多空诊断 | 20bp | 20.62% | 15.83% | 1.302 | -21.93% | 113.17% |
| 沪深300 | 0bp | 1.01% | 18.39% | 0.055 | -39.92% | — |

最终投资判断必须同时参考 `output/alpha_diagnostics.csv`、`output/advanced_cost_stress.csv`、`output/advanced_subperiod.csv` 和 `output/capacity_analysis.csv`，不能只引用上表中最好的一个数字。

截面统计显示平均月度 Rank IC 为0.117，年化 ICIR 为3.07，87.37%的月份 IC 为正；十分组平均收益与分组序号的 Spearman 相关系数为0.952，行业中性多空月收益 t 值为3.54。多空组合在2018–2021和2022–2025的20bp净夏普分别为1.56和1.12，方向没有在后半段消失。

但结果有三个不能回避的限制：

1. 多头组合夏普仍只有0.565，不能达到独立策略的可投标准；
2. 多空组合月均换手为113.17%，成本从20bp提高至50bp后夏普降至1.00，100bp时只剩0.52，200bp时转负；
3. 非线性模型只在25%的重估点通过历史准入门槛，完整样本的最终 IC 反而比纯 Ridge 低0.0039，说明当前 TabM/检索模型没有提供可靠的样本外增量。

因此本版的客观结论是：数据中存在值得继续验证的行业内截面排序信号，但“可交易多头”仍不可投，“行业中性多空”也只是未计融券和真实冲击的研究上界，不具备直接带资上线条件。

## 7. 运行与复核

```powershell
python -m pip install -r requirements.txt
python quant_project.py --start 2014-01-01 --end 2025-12-31
python advanced_quant.py --start 2018-01-01
python advanced_robustness.py
python model_comparison_quant.py
python turnover_aware_quant.py
python turnover_aware_robustness.py
python -m unittest -q test_advanced_quant.py
python -m unittest -q test_model_comparison_quant.py
python -m unittest -q test_turnover_aware_quant.py
```

主要输出：

- `output/advanced_backtest.csv`：月度收益、换手、净值、IC和模型权重；
- `output/advanced_metrics.csv`：多头、多空诊断、旧模型和基准汇总；
- `output/alpha_diagnostics.csv`：截面 Alpha 统计；
- `output/decile_returns.csv`：行业内十分组月收益；
- `output/advanced_model_log.csv`：每次重估的验证 IC、模型准入和权重；
- `output/linear_factor_coefficients.csv`：线性系数历史；
- `output/advanced_cost_stress.csv`：0至200bp成本压力；
- `output/advanced_subperiod.csv`：2018–2021和2022–2025分段表现；
- `output/capacity_analysis.csv`：不同资金规模下的ADV参与率近似。

## 8. 当前研究边界

本项目仍缺少逐笔/盘口数据、涨跌停排队成交、融券可得性与借券费、指数成分和行业风险模型、真实冲击函数以及实盘订单回报。现有容量表也只覆盖多头组合，不能外推到融券组合。因此当前结果最多证明“月频截面信号值得继续研究”，不能证明“策略可以带资上线”。下一步优先事项是降低行业中性多空换手、做完全隔离的滚动留出期验证，并用真实可成交价格和融券池重估净 Alpha；继续堆叠更复杂网络不是优先事项。

## 9. 未来信息审计与换手约束版本

`turnover_aware_quant.py` 是独立于稳定版 `advanced_quant.py` 的纯 Ridge 路径，不增加新因子，也不让 TabM 或检索模型进入正式信号。换手控制由进入/退出缓冲、预期 Alpha 覆盖成本门槛和每3个月一次计划调仓构成；非计划调仓月只处理失去资格的持仓与必要补位。

### 发现并修复的未来信息问题

审计发现旧换手版本在当月打分股票池上附加了 `forward_return.notna()`。这等于提前知道某只股票下月是否存在收益记录，属于未来可用性泄漏。现已修正：

- 当月股票池只由当月 `eligible` 状态决定，不再用下一月收益是否存在筛选；
- 全局最后信号月固定为2025-11，不再按单只股票的未来记录决定边界；
- 缺失下一月收益的股票仍被打分，组合输出单独记录缺失收益权重；
- 财务指标按公告日向后匹配，滚动因子只使用当月及历史，训练和验证月份严格早于信号月；
- `output/future_leakage_audit.csv` 保存可机器复核的时间边界。完整样本有5个“已打分但下一月收益缺失”的股票月份，证明打分池没有再被未来收益可用性过滤。

组合收益暂将缺失的个股下月收益记为零贡献，并在 `missing_forward_return_weight` 字段披露；这不是退市收益模型，也不能替代真实退市、停牌和复牌价格处理。本次持仓中的最大缺失权重约0.26%，仍属于必须继续改进的数据边界。

### 时间分区与冻结规则

- formation：2014-01 至 2017-12，只用于既有因子方向审计；
- development：2018-01 至 2021-12；
- selection：2022-01 至 2023-12；
- retrospective_test：2024-01 至 2025-12；由于最后一个可实现标签来自2025-11信号，实际回顾性收益有23个月；
- true_forward_start：2026-01-01。

2018—2025已经被反复查看，因此2024—2025只能称为回顾性测试，不能重新包装成完全未见样本。18组参数网格只接收 development 和 selection 行；选定参数后才运行一次 retrospective_test。最终配置、区间、随机种子、代码SHA-256、运行时间和Git提交保存在 `output/research_lock.json`。2026年以后才是冻结参数后的真正前向验证。

为完整复现2014—2017 formation，月度面板需要从2014开始生成。任务指定的2016起始命令已经做过运行回归，但它本身不足以覆盖完整formation：

```powershell
python quant_project.py --start 2014-01-01 --end 2025-12-31
python advanced_quant.py --start 2018-01-01
python turnover_aware_quant.py
python turnover_aware_robustness.py
python model_comparison_quant.py
python -m unittest -q test_advanced_quant.py
python -m unittest -q test_turnover_aware_quant.py
python -m unittest -q test_model_comparison_quant.py
```

### 规则选择结果

只比较了预先声明的小网格：`smoothing_weight ∈ {1.00, 0.75, 0.50}`、`exit_fraction ∈ {0.20, 0.25}`、`hurdle_multiple ∈ {1.0, 1.5, 2.0}`。selection期综合50bp净Sharpe、换手、IC、分组单调性和相邻参数稳定性后，冻结参数为：

- `smoothing_weight = 1.00`；
- `exit_fraction = 0.25`；
- `hurdle_multiple = 1.0`；
- `rebalance_frequency_months = 3`；
- 5日执行、5000万元默认组合规模、单只股票最高10%五日ADV参与率；
- 基础成本20bp，冲击代理系数0.10，根号内参与率截断至 `[0, 0.25]`。

`smoothing_weight = 1.00` 表明数据没有支持额外时间平滑。selection期多空换手相对稳定版下降65.26%，Rank IC不下降，十分组单调性为1.0；3个相邻候选全部通过开发/选择期约束，因此结果不依赖唯一一个精确参数。回顾期没有参与这项稳定性判断。

### 严格同月份比较

稳定版多空序列有两个月因个别成分下月收益缺失而为空。为避免95个月与93个月混比，下表的收益、Sharpe、回撤和多空换手统一使用双方均非空的93个月；多头比较使用共同95个月。

| 指标 | 当前高级版本 | 换手约束Ridge | 变化 |
|---|---:|---:|---:|
| 平均月度Rank IC | 0.1171 | 0.1210 | +0.0039 |
| 年化ICIR | 3.071 | 3.245 | +0.174 |
| 正IC月份比例 | 87.37% | 85.26% | -2.11个百分点 |
| 十分组单调性 | 0.952 | 0.964 | +0.012 |
| 多空20bp月收益t值 | 3.54 | 2.45 | -1.09 |
| 多头月均换手 | 34.86% | 14.94% | **下降57.13%** |
| 多空月均换手 | 113.38% | 37.64% | **下降66.80%** |

| 多空诊断（共同93个月） | 当前高级版本 | 换手约束Ridge | 变化 |
|---|---:|---:|---:|
| 20bp年化收益 | 20.62% | 12.40% | -8.22个百分点 |
| 20bp Sharpe | 1.302 | 0.850 | -0.452 |
| 50bp Sharpe | **1.000** | **0.745** | **-0.255** |
| 100bp Sharpe | 0.521 | 0.573 | +0.052 |
| 20bp最大回撤 | -21.93% | -23.88% | -1.95个百分点 |

换手显著下降，但20bp收益牺牲过多，50bp也没有改善；只有100bp高成本情景略好。不能把“更低换手”等同于“策略收益提高”。

### 分时期结果

下表是换手约束多空诊断组合，全部为固定20bp成本：

| 阶段 | 年化收益 | Sharpe | 最大回撤 | 月均换手 |
|---|---:|---:|---:|---:|
| development | 14.64% | 1.246 | -10.96% | 39.25% |
| selection | 22.90% | 1.749 | -4.25% | 37.57% |
| retrospective_test | **0.46%** | **0.023** | **-22.74%** | 35.10% |

retrospective_test在20bp下勉强保持正收益，但50bp年化收益为-0.80%、Sharpe为-0.040。selection到回顾期的明显衰减说明历史选择结果不能外推为可投资收益。

换手约束多头组合在20bp下年化9.05%、Sharpe 0.480、最大回撤-26.58%，仍不达到独立策略的可投门槛。其回顾期表现较强，但selection期20bp收益接近零，稳定性不足。

### 成本与容量代理

冲击代理使用月内平均日成交额、年化波动率、5日执行和订单金额：

`base_cost + impact_coefficient × volatility × sqrt(clipped(order_value / (ADV × 5)))`

它只是流动性分层代理，不是真实市场冲击函数。按该代理，5000万元多头组合Sharpe为0.446，多空诊断Sharpe为0.754；后者仍未计融券可得性、借券费和基差。多头交易在1亿元规模下五日ADV参与率p99约0.35%，但该数字依赖 `amount_clean` 作为ADV近似，不能视为正式容量证明。

### 最终验收结论

数据中仍存在行业内截面排序能力，未来收益可用性泄漏已经修复。季度调仓把多空月均换手降低约66.8%，但没有改善20bp或50bp净表现，回顾期50bp仍为负。因此本版本**验收失败，不能升级为生产候选**。机器学习对照也没有稳定优于普通线性回归，本轮到此停止增加模型或因子。

行业中性多空仍只是未模拟融券约束的研究上界。`research_lock.json` 冻结的是研究配置，不代表策略已可投；2026年以后必须在不改参数的前提下积累真正前向结果。

新增输出：

- `output/future_leakage_audit.csv`：未来信息检查与证据；
- `output/model_comparison_monthly.csv`、`output/model_comparison_metrics.csv`：清洗后OLS、Ridge、Huber、TabM和检索模型对比；
- `output/turnover_aware_backtest.csv`：逐月信号、实现月份、IC、计划调仓标记、缺失收益权重和多空两侧换手；
- `output/turnover_aware_metrics.csv`：20bp与流动性代理成本汇总；
- `output/turnover_aware_cost_stress.csv`：与当前版本的0至200bp同口径压力测试；
- `output/turnover_aware_subperiod.csv`：三个研究阶段在20/50/100bp下的表现；
- `output/turnover_aware_trade_log.csv`：实际与被拒绝交易、预期Alpha、成本和ADV参与率；
- `output/turnover_aware_capacity.csv`：1000万、5000万和1亿元多头容量代理；
- `output/turnover_aware_parameter_selection.csv`：仅development/selection参数比较；
- `output/turnover_aware_decile_returns.csv`：行业内十分组收益；
- `output/turnover_aware_alpha_diagnostics.csv`：与当前高级版本的核心指标对比；
- `output/research_lock.json`：2026前向验证冻结配置。
