# A股多因子：严格清洗、MLP截面排序与低换手组合

项目保留原来的MLP与岭回归动态集成作为第一版可复现基准，并新增严格清洗的低换手优化版。新版本先运行普通Ridge回归，再将Ridge、MLP及动态融合放在同一时间序列框架中比较；正式候选不是因为2024—2025表现更好而选出，而是由2018—2021开发期和2022—2023选择期共同确定。

旧版曾报告年化10.83%、Sharpe 0.569、最大回撤-24.97%。在当前原始数据状态下重新生成2016—2025月度面板、重新执行异常值和缺失值处理后，可复现结果为年化10.58%、Sharpe 0.575、最大回撤-24.19%。旧数字只作为历史参照，当前输出文件中的新数字才是正式结果。

## 当前优化结果（2026-08冻结）

冻结配置为：清洗后MLP信号、`smoothing_weight=0.75`、`exit_fraction=0.20`、前10%等权多头。模型最长使用60个月训练数据，最后12个月作为历史验证窗口，每季度重新估计。每个预测月只能使用当月末及以前已经实现的标签。

| 20bp口径 | 年化收益 | Sharpe | 最大回撤 | 月均换手 |
|---|---:|---:|---:|---:|
| **清洗后低换手MLP** | **11.59%** | **0.627** | -24.64% | **25.78%** |
| 第一版MLP＋Ridge | 10.58% | 0.575 | **-24.19%** | 38.36% |

新版本相对第一版：年化收益提高约1.01个百分点，Sharpe提高0.052，月均换手降低约32.8%；代价是最大回撤恶化约0.45个百分点。因此结论是收益、Sharpe和成本敏感度改善，但回撤没有全面改善。

| 单边成本 | 新版Sharpe | 第一版Sharpe |
|---:|---:|---:|
| 20bp | **0.627** | 0.575 |
| 50bp | **0.572** | 0.494 |
| 100bp | **0.480** | 0.360 |

波动率控制版本也被实际测试，但其20bp Sharpe只有0.488，低于完全投资版本，因此不进入正式候选。没有因为风险控制“听起来合理”就把它写成成功结果。

### 2026-08-12 Sharpe稳健性复核

本轮没有修改冻结生产配置。研究只使用已有清洗因子、Ridge/MLP走步预测和小型组合网格，比较标准以50bp成本下的development与selection共同表现为主；2024—2025结果单独写入回顾文件，不参与候选准入。

| 候选 | development Sharpe（50bp） | selection Sharpe（50bp） | 全期Sharpe（20bp） | 全期Sharpe（50bp） | 月均换手 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| **冻结版：前10%、平滑0.75、退出20%** | **0.550** | 0.236 | 0.627 | **0.572** | **25.78%** | 继续保留 |
| 前5%集中组合 | 0.474 | **0.357** | **0.634** | 0.549 | 40.50% | 拒绝：换手和成本后表现恶化 |
| 分数排名轻度倾斜 | 0.541 | 0.248 | 0.629 | 0.569 | 28.31% | 拒绝：增益不足且换手上升 |
| 剔除波动最高10% | 0.548 | 0.293 | 0.632 | 0.572 | 27.68% | 拒绝：50bp增益仅0.0005且换手上升 |
| Ridge/MLP共识排序 | 0.509 | 0.226 | 0.596 | 0.533 | 29.58% | 拒绝：跨阶段退化 |
| 10%价值轻度校准 | 0.476 | 0.249 | 0.611 | 0.558 | 24.60% | 拒绝：development明显退化 |

还检查了36/60/84个月训练窗口、质量与低波校准、信号分散度降仓以及固定比例的核心/防御组合。60个月窗口仍是开发期与选择期较稳健的选择；其余方案未同时改善两个阶段。完整固定候选表见`output/sharpe_candidate_selection.csv`，成本压力见`output/sharpe_candidate_cost_stress.csv`，回顾结果见`output/sharpe_candidate_retrospective.csv`。

结论不是“夏普已被显著提高”，而是：在不引入新数据、不使用2024—2025挑参数、并把真实权重变化计入成本的约束下，没有找到可替代Sharpe 0.627冻结版的稳健候选。前5%方案的0.634只是全样本20bp口径上的小幅提高，不能抵消换手从25.78%升至40.50%和50bp Sharpe下降。继续扩大参数搜索只会加重过拟合，因此本轮把失败候选作为可复现稳健性结果，而不改写生产结论。

### CSI 500股指期货对冲代理

`risk_overlay_quant.py`使用滞后12个月滚动beta和50%对冲比例做独立诊断，并将期货名义敞口变化、交易成本和年化展期成本计入收益。现金指数收益只作为期货收益代理，没有模拟基差、保证金和合约换月，因此不能称为真实期货回测。全期20bp股票成本口径下，对冲代理Sharpe为0.619，低于未对冲的0.627；它降低了部分阶段波动，却牺牲了收益，因此不进入正式组合。

### 独立数据实验：卖方研报与盈利预测

本轮新增了独立于CSMAR面板的东方财富历史个股研报快照。`external_data_quant.py`按年度、逐页缓存2017—2025数据，共取得139,937份研报、覆盖4,688只股票。每个股票月只读取不晚于当月末发布的研报，并在每次季度重估中同时检查标签实现月份和研报发布日期。原始研报缓存位于`data/`，不提交GitHub。

构造的点时特征包括近90日研报数量、近180日机构数、分析师评级、EPS预期增长和三个月预期修正。2018—2023合格股票的月均近90日研报覆盖率为26.54%，点时审计全部通过。但分阶段IC显示明显不稳定：

| 外部特征 | development Rank IC | selection Rank IC | 结论 |
|---|---:|---:|---|
| 研报数量 | 0.0506 | -0.0213 | 方向反转 |
| 机构覆盖数 | 0.0509 | -0.0313 | 方向反转 |
| 分析师评级 | 0.0082 | 0.0028 | 同号但很弱 |
| EPS预期增长 | -0.0020 | -0.0242 | 覆盖月份不足 |
| EPS预期修正 | -0.0035 | -0.0181 | 覆盖月份不足 |

把五个外部特征直接加入Ridge/MLP后，20bp全期年化收益为10.19%、Sharpe 0.556、最大回撤-23.38%、月均换手27.49%，低于冻结版的Sharpe 0.627。进一步只对跨阶段同号的分析师评级测试2.5%/5%/10%轻量叠加，最小的2.5%权重仍使selection 50bp Sharpe从0.236降至0.222。因此本轮独立数据接入在工程和点时审计上成功，但没有形成可接受的增量Alpha，正式生产配置继续保持不变。

北向资金个股持仓也被评估为候选数据源，但当前公开接口只能稳定提供近期数据，无法覆盖2018—2023的开发/选择区间，因此未把它拼入历史回测。

### 时间区间与结果解释

- development：2018-02至2021-12，用于开发；
- selection：2022-01至2023-12，用于选择模型、平滑权重和退出缓冲；
- retrospective_test：2024-01至2025-12，只在参数冻结后输出，但这些年份在过去研究中已经被查看，不能称为完全未见样本；
- true forward：本次配置于2026-08重新冻结，最早从2026-09实现收益开始才属于冻结后的真正前向验证。

2024—2025回顾性测试中，新版年化收益20.63%、Sharpe 0.958；第一版同期年化20.07%、Sharpe 0.970。新版收益略高但同期Sharpe略低，说明全样本Sharpe提升并不是2024—2025单一阶段全面占优造成的，也不能把这两年当作未来业绩证明。

所选平滑与缓冲并非孤立尖点：相邻的MLP参数组合在开发期和选择期的50bp净收益均为正，完整比较见`output/optimized_parameter_selection.csv`。

## 1. 项目目标

模型只使用月末已经知道的信息。第一版预测股票下一月在全市场中的相对收益分位数；优化版进一步把训练目标改为下一月行业内收益分位数，使模型学习行业内截面排序而不是行业涨跌。两版的原始因子都先在月度行业截面内转换为排名。

研究流程：

```text
CSMAR原始压缩数据
    ↓
2016—2025月度行情、估值、财务公告和行业数据
    ↓
异常范围过滤、缺失标记、行业内排名
    ↓
非线性多因子岭回归基线
    ↓
MLP与岭回归历史验证期动态加权
    ↓
前10%多头组合、15%退出缓冲
    ↓
成本压力与未来函数审计
```

## 2. 数据清洗

所有回归和MLP训练之前执行：

- 股票代码统一为六位并按股票—月份去重；
- 月收益限制在 `[-95%, 300%]` 的经济范围；
- PE、PB、PS只保留正且不过度极端的值；
- 年化波动率限制在 `[3%, 300%]`；
- ROE、ROA、毛利率、现金转化率和资产负债率执行范围检查；
- 非正成交额、市值和负非流动性数据不进入有效特征；
- 每个因子在月度行业截面内转成分位数；
- 缺失值不使用全样本信息填充，而是设为中性值并加入独立缺失标记；
- 股票需上市至少180天、当月交易不少于10天，并位于成交额前80%；
- 财务指标由基础面板按照公告日向后匹配，不能在公告前使用。

每个特征的缺失数量和最终合格样本比例保存在`output/deep_learning_data_quality.csv`。

## 3. 模型结构

### 岭回归基线

`ml_quant.py`使用13个基础因子、平方项和少量预先固定的交互项。正则化参数只在历史训练窗口末尾12个月中选择。

### MLP

`deep_learning_quant.py`实现两层前馈网络：

- 输入：13个行业排名因子及13个缺失标记；
- 隐藏层：32和16个ReLU神经元；
- 损失：均方误差加L2约束；
- 优化：Adam；
- 提前停止：只观察历史验证集；
- 固定随机种子，保证可复现。

每个季度分别计算MLP和岭回归在历史验证期的Rank IC，然后只在以下固定权重中选择：

```text
MLP权重 = 0%, 25%, 50%, 75%, 100%
```

没有使用连续权重搜索，也没有根据最终回测期逐月挑选模型。

## 4. 时间边界

- 月度面板：2016-01至2025-12；
- 样本外信号：2018-01至2025-11；
- 实现收益：2018-02至2025-12，共95个月；
- 训练窗口：最长60个月；
- 历史验证窗口：训练窗口末尾12个月；
- 模型重估：每年1月、4月、7月和10月；
- 组合：预测最高10%的股票等权持有；
- 退出缓冲：已有持仓跌出前15%后退出；
- 默认单边成本：20bp。

训练样本月份严格早于预测月份。最晚训练标签必须在预测月末之前已经实现，否则程序直接报错。本次全部重估点均通过`output/deep_learning_leakage_audit.csv`中的检查。

## 5. 当前正式结果

### 20bp成本

| 策略 | 年化收益 | 年化波动 | Sharpe | 最大回撤 | 月数 |
|---|---:|---:|---:|---:|---:|
| MLP＋岭回归动态集成 | **10.58%** | **18.38%** | **0.575** | **-24.19%** | 95 |
| 非线性岭回归基线 | 9.85% | 19.06% | 0.517 | -24.73% | 95 |
| 沪深300 | 1.01% | 18.39% | 0.055 | -39.92% | 95 |

相对岭回归，MLP动态集成：

- 年化收益提高约0.72个百分点；
- Sharpe提高约0.058；
- 最大回撤改善约0.53个百分点；
- 收益、风险调整表现和回撤方向一致。

### 成本压力

| 单边成本 | 年化收益 | Sharpe | 最大回撤 |
|---:|---:|---:|---:|
| 0bp | 11.59% | 0.630 | -23.85% |
| 20bp | 10.58% | 0.575 | -24.19% |
| 50bp | 9.07% | 0.494 | -24.77% |
| 100bp | **6.60%** | **0.360** | -25.73% |
| 150bp | 4.19% | 0.228 | -26.67% |
| 200bp | 1.82% | 0.099 | -27.80% |

100bp成本下仍为正收益，但Sharpe只有0.360，说明策略对交易成本依然敏感，不能据此宣称已经达到实盘可投标准。

## 6. 历史第一版与当前复跑的区别

| 指标 | 历史输出 | 当前清洗复跑 |
|---|---:|---:|
| 年化收益 | 10.83% | 10.58% |
| Sharpe | 0.569 | 0.575 |
| 最大回撤 | -24.97% | -24.19% |
| 100bp年化收益 | 5.94% | 6.60% |

当前版本没有直接复制旧CSV。所有正式指标均由当前代码重新运行生成。历史数字变化来自月度面板重新构建、股票代码去重、异常收益边界、非流动性边界和缺失标记的统一处理。

## 7. 运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe quant_project.py --start 2016-01-01 --end 2025-12-31
.\.venv\Scripts\python.exe ml_quant.py --start 2018-01-01
.\.venv\Scripts\python.exe deep_learning_quant.py --start 2018-01-01
.\.venv\Scripts\python.exe optimized_quant.py
.\.venv\Scripts\python.exe sharpe_robustness.py
.\.venv\Scripts\python.exe risk_overlay_quant.py
.\.venv\Scripts\python.exe external_data_quant.py --refresh

.\.venv\Scripts\python.exe -m unittest -q test_data_cleaning.py
.\.venv\Scripts\python.exe -m unittest -q test_deep_learning_quant.py
.\.venv\Scripts\python.exe -m unittest -q test_optimized_quant.py
.\.venv\Scripts\python.exe -m unittest -q test_sharpe_robustness.py
.\.venv\Scripts\python.exe -m unittest -q test_risk_overlay_quant.py
.\.venv\Scripts\python.exe -m unittest -q test_external_data_quant.py
```

原始CSMAR数据受许可限制，不上传GitHub。运行前需将对应压缩文件放入`data/`目录。

## 8. 正式输出

- `output/monthly_panel.csv.gz`：本地重建月度面板，不提交GitHub；
- `output/backtest_monthly.csv`、`output/metrics.csv`：透明基础多因子结果；
- `output/ml_backtest_monthly.csv`、`output/ml_metrics.csv`：岭回归基线；
- `output/deep_learning_backtest.csv`：MLP动态集成逐月收益；
- `output/deep_learning_metrics.csv`：正式指标；
- `output/deep_learning_model_log.csv`：每次重估的验证IC、训练轮数和MLP权重；
- `output/deep_learning_leakage_audit.csv`：未来函数审计；
- `output/deep_learning_cost_stress.csv`：0至200bp成本压力；
- `output/deep_learning_data_quality.csv`：模型输入清洗报告；
- `output/factor_ic.csv`：基础因子IC。
- `output/optimized_backtest.csv`：冻结低换手候选的逐月收益、真实权重换手和风险控制对照；
- `output/optimized_metrics.csv`、`output/optimized_comparison.csv`：新版、第一版和基准的同口径比较；
- `output/optimized_cost_stress.csv`：0至200bp成本压力；
- `output/optimized_parameter_selection.csv`：仅使用开发期和选择期的27组小网格；
- `output/optimized_subperiod.csv`：开发、选择和回顾测试分阶段结果；
- `output/optimized_alpha_diagnostics.csv`、`output/optimized_decile_returns.csv`：月度Rank IC与十分组单调性；
- `output/optimized_model_log.csv`、`output/optimized_leakage_audit.csv`：季度重估和标签实现日期审计；
- `output/optimized_research_lock.json`：冻结参数、时间边界和代码SHA-256。
- `output/sharpe_candidate_selection.csv`：仅用development和selection生成的候选准入表；
- `output/sharpe_candidate_cost_stress.csv`：本轮固定候选在20/50/100bp下的全期压力结果；
- `output/sharpe_candidate_retrospective.csv`：与选择表物理分离的2024—2025回顾结果；
- `output/risk_overlay_backtest.csv`、`output/risk_overlay_metrics.csv`、`output/risk_overlay_cost_stress.csv`：CSI 500期货对冲代理诊断。
- `output/external_data_quality.csv`：独立研报数据的日期、报告数、股票数、覆盖率与点时审计；
- `output/external_feature_ic.csv`：五个外部特征在development、selection与retrospective_test的分阶段Rank IC；
- `output/external_backtest.csv`、`output/external_subperiod.csv`、`output/external_cost_stress.csv`：外部特征增强模型结果；
- `output/external_parameter_selection.csv`、`output/external_leakage_audit.csv`、`output/external_research_lock.json`：参数选择、未来数据审计与冻结记录；
- `output/external_rating_overlay_selection.csv`：分析师评级2.5%/5%/10%轻量叠加对照。

## 9. 限制

本项目没有逐笔订单簿、涨跌停排队成交、真实冲击函数、历史指数成分权重或实盘订单回报。20—200bp只是固定成本压力，不等于真实冲击模型。当前结果仍是月频历史研究结果；Sharpe 0.627尚不足以证明达到机构实盘可投门槛，不构成投资建议，也不代表未来收益。
