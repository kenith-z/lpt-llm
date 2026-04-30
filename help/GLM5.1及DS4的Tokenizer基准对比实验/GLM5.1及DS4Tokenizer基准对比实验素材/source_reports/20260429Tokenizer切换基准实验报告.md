# 20260429 Tokenizer 切换基准实验报告

## 1. 实验目的

本次实验用于评估 `glm_tokenizer` 切换到 `ds_tokenizer` 后，对专升本教材语料 token 长度分布的影响。

上一轮只使用 `大学语文` 单个 JSONL，样本数为 `38`，代表性偏弱。本轮扩大到编号 `1-11` 的专升本考试专用教材，共 `11` 个 JSONL 文件，以获得更稳定的数据判断。

重点回答三件事：

1. 切换到 `ds_tokenizer` 后，总 token 规模如何变化。
2. 单样本长度分布是否显著改变。
3. 当前 `train_max_sequence_length=7680` 下，样本截断风险是否变化。

## 2. 实验对象

本次纳入以下 `11` 个文件：

1. `1--2023版河南省普通高校专升本考试专用教材·动物、植物遗传学_parsed_output.text.jsonl`
2. `2--2023年河南省普通高校专升本考试专用教材·经济学（用2022年加印）_parsed_output.text.jsonl`
3. `3--2023年广东省普通高校专升本考试专用教材·教育理论_parsed_output.text.jsonl`
4. `4--2023年广东省普通高校专升本考试专用教材·英语_parsed_output.text.jsonl`
5. `5---2023年广东省普通高校专升本考试专用教材·生理学- 转曲_parsed_output.text.jsonl`
6. `6--2023年广东省普通高校专升本考试专用教材·管理学_parsed_output.text.jsonl`
7. `7--2023年广东省普通高校专升本考试专用教材·高等数学_parsed_output.text.jsonl`
8. `8--2023年广东省普通高校专升本考试专用教材·大学语文_parsed_output.text.jsonl`
9. `9--2023年广东省普通高校专升本考试专用教材·艺术概论_parsed_output.text.jsonl`
10. `10--2023年广东省普通高校专升本考试专用教材·政治理论_parsed_output.text.jsonl`
11. `11--2023年广东省普通高校专升本考试专用教材·民法_parsed_output.text.jsonl`

总体数据：

- 文件数：`11`
- 有效样本数：`416`
- 样本类型：`text`
- 字符总量：`4164867`
- 平均字符数：`10011.70`
- 中位数字符数：`10173`
- 最短样本：`454` 字符
- 最长样本：`10239` 字符

说明：

- 本次统计使用 JSONL 记录中的 `text` 字段。
- 统计命令使用 `add_special_tokens=False`。
- 实际训练时 text 样本会额外追加模板定义的 EOS token，因此训练序列长度通常比本报告中的正文 token 数多 `1`。

## 3. 实验环境

- 日期：2026-04-29
- 操作系统：Windows 10
- Python 环境：项目 `.venv`
- 对比脚本：`tests/test_tokenizer_jsonl_token_counts.py`
- GLM tokenizer：`lpt_model/glm_tokenizer`
- DS tokenizer：`lpt_model/ds_tokenizer`

## 4. 实验方法

### 4.1 脚本改造

`tests/test_tokenizer_jsonl_token_counts.py` 已从单文件输入扩展为多文件输入：

- CLI 支持一次传入多个 JSONL 路径。
- 控制台逐样本输出增加 `source` 文件名。
- `TOKENIZER_COMPARE_JSONL` 环境变量模式支持用系统路径分隔符传多个文件。
- tokenizer 只加载一次，避免多文件统计时重复加载。

### 4.2 使用命令

```powershell
$paths = Get-ChildItem -LiteralPath .\data\structured -Filter '*.text.jsonl' |
  Where-Object { $_.Name -match '^(?:[1-9]|10|11)-+.*专升本考试专用教材' } |
  Sort-Object { [int]([regex]::Match($_.Name, '^\d+').Value) }

.\.venv\Scripts\python.exe tests\test_tokenizer_jsonl_token_counts.py @($paths.FullName)
```

### 4.3 统计口径

脚本逐行读取 JSONL，并分别用两种 tokenizer 编码正文：

```python
len(tokenizer(text, add_special_tokens=False)["input_ids"])
```

输出字段：

- `source`：JSONL 文件名
- `line`：JSONL 行号
- `id`：样本 ID
- `type`：样本类型
- `chars`：正文字符数
- `glm_tokens`：GLM tokenizer token 数
- `ds_tokens`：DS tokenizer token 数
- `ds-glm`：`ds_tokens - glm_tokens`

## 5. 基准结果

### 5.1 总体结果

| 指标 | GLM tokenizer | DS tokenizer | 变化 |
|---|---:|---:|---:|
| 总 token 数 | 2357420 | 2226598 | -130822 |
| 平均 token 数 | 5666.88 | 5352.40 | -314.48 |
| 中位数 token 数 | 5778.50 | 5522.00 | -256.50 |
| 最小 token 数 | 246 | 240 | -6 |
| P90 token 数 | 6562 | 6340 | -222 |
| P95 token 数 | 6913 | 6784 | -129 |
| P99 token 数 | 7975 | 7787 | -188 |
| 最大 token 数 | 9119 | 8770 | -349 |

总量对比：

- `DS / GLM = 0.944506`
- DS 总 token 规模下降约 `5.5494%`

### 5.2 单样本差异范围

| 指标 | 数值 |
|---|---:|
| 平均差异 `ds-glm` | -314.48 |
| 中位数差异 `ds-glm` | -209.00 |
| 最小差异 `ds-glm` | -1288 |
| 最大差异 `ds-glm` | 33 |
| 平均 `DS/GLM` 比例 | 0.943902 |
| 最低 `DS/GLM` 比例 | 0.749903 |
| 最高 `DS/GLM` 比例 | 1.017259 |

观察：

- `416` 条样本中，`415` 条 DS token 数低于 GLM。
- 只有 `1` 条样本 DS 比 GLM 多 `33` token：
  - `4--2023年广东省普通高校专升本考试专用教材·英语...`
  - line `50`
  - GLM `1912`
  - DS `1945`

### 5.3 分教材结果

| 编号 | 教材 | 样本数 | GLM tokens | DS tokens | 变化 | 降幅 | DS 最大 | DS > 7680 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 动物、植物遗传学 | 25 | 152414 | 144643 | -7771 | 5.0986% | 6401 | 0 |
| 2 | 经济学 | 43 | 248416 | 235666 | -12750 | 5.1325% | 5859 | 0 |
| 3 | 教育理论 | 28 | 154193 | 150089 | -4104 | 2.6616% | 6041 | 0 |
| 4 | 英语 | 50 | 197731 | 191385 | -6346 | 3.2094% | 5010 | 0 |
| 5 | 生理学 | 32 | 198736 | 188894 | -9842 | 4.9523% | 6403 | 0 |
| 6 | 管理学 | 31 | 176079 | 171607 | -4472 | 2.5398% | 5865 | 0 |
| 7 | 高等数学 | 65 | 355023 | 296792 | -58231 | 16.4020% | 5571 | 0 |
| 8 | 大学语文 | 38 | 269514 | 261881 | -7633 | 2.8321% | 8770 | 5 |
| 9 | 艺术概论 | 34 | 205111 | 199538 | -5573 | 2.7171% | 6474 | 0 |
| 10 | 政治理论 | 36 | 191743 | 185539 | -6204 | 3.2356% | 5823 | 0 |
| 11 | 民法 | 34 | 208460 | 200564 | -7896 | 3.7878% | 6419 | 0 |

关键观察：

- `高等数学` 的降幅最大，达到 `16.4020%`。
- `大学语文` 的降幅不大，但它是唯一仍存在 `DS > 7680` 样本的教材。
- 其他 10 本教材在 DS 下最大样本均不超过 `6474` token。

### 5.4 长度阈值影响

当前项目默认训练长度上限为 `GlobalConfig.train_max_sequence_length=7680`。

| 阈值 | GLM 超限样本数 | DS 超限样本数 | 变化 |
|---|---:|---:|---:|
| > 7680 | 10 | 5 | -5 |
| > 8192 | 3 | 1 | -2 |
| > 8960 | 1 | 0 | -1 |
| > 10240 | 0 | 0 | 0 |

说明：

- `>7680` 的 DS 超限样本全部来自 `大学语文`。
- DS 下最大正文长度为 `8770` token。
- 训练时加 EOS 后，最大训练序列约为 `8771` token。

### 5.5 从超限降为不超限的样本

以下样本在 GLM 下超过 `7680`，但在 DS 下回落到 `7680` 以内：

| source | line | GLM tokens | DS tokens | ds-glm |
|---|---:|---:|---:|---:|
| 大学语文 | 5 | 7845 | 7633 | -212 |
| 大学语文 | 27 | 7722 | 7521 | -201 |
| 大学语文 | 29 | 7840 | 7641 | -199 |
| 大学语文 | 31 | 7746 | 7539 | -207 |
| 大学语文 | 34 | 7749 | 7559 | -190 |

这些样本是 tokenizer 切换对 `7680` 训练窗口最直接的收益点。

### 5.6 DS 下仍超过 7680 的样本

| source | line | GLM tokens | DS tokens | ds-glm |
|---|---:|---:|---:|---:|
| 大学语文 | 4 | 8060 | 7787 | -273 |
| 大学语文 | 14 | 8261 | 8023 | -238 |
| 大学语文 | 35 | 7975 | 7797 | -178 |
| 大学语文 | 36 | 9119 | 8770 | -349 |
| 大学语文 | 37 | 8222 | 7883 | -339 |

### 5.7 token 降幅最大的样本

降幅最大的前 10 条全部来自 `高等数学`：

| line | GLM tokens | DS tokens | ds-glm |
|---:|---:|---:|---:|
| 60 | 5150 | 3862 | -1288 |
| 58 | 5421 | 4154 | -1267 |
| 59 | 5439 | 4276 | -1163 |
| 36 | 5412 | 4292 | -1120 |
| 40 | 5297 | 4198 | -1099 |
| 32 | 5158 | 4066 | -1092 |
| 38 | 5753 | 4675 | -1078 |
| 15 | 5404 | 4332 | -1072 |
| 35 | 5338 | 4287 | -1051 |
| 43 | 5645 | 4598 | -1047 |

这说明 DS tokenizer 对数学教材中的符号、公式或数字结构更友好，是本次总 token 降幅从单文件 `2.83%` 扩大到全样本 `5.55%` 的主要原因。

## 6. 结果分析

### 6.1 扩大样本后，DS token 降幅更可信

单个 `大学语文` 文件上，DS 总 token 降幅约 `2.83%`。

扩展到 11 本教材后，DS 总 token 降幅提升到 `5.55%`。这说明单文件结果低估了 tokenizer 切换收益，尤其没有覆盖数学类材料中更明显的压缩效果。

### 6.2 截断风险仍由少数长样本主导

虽然整体 token 数下降明显，但 `7680` 训练窗口的超限样本仍全部集中在 `大学语文`。

这意味着：

- 大部分教材在当前窗口下已经安全。
- 是否提高训练窗口，主要取决于是否要完整保留 `大学语文` 的少数长样本。

### 6.3 对训练配置的含义

如果目标是保持当前显存预算，并减少截断：

- `train_max_sequence_length=7680`
- `ds_tokenizer`

这是合理配置。相比 GLM，DS 已把超限样本从 `10` 条降到 `5` 条。

如果目标是让这 11 本教材全部样本完整进入训练窗口：

- 建议评估 `train_max_sequence_length=8960`

理由：

- DS 最大正文长度为 `8770`
- text 样本训练时追加 EOS 后约 `8771`
- `8960` 可覆盖当前 11 本教材，并保留少量余量

### 6.4 对 sequence packing 的影响

DS 总 token 下降 `130822`，会直接降低训练总 token 规模。

更重要的是：

- 数学教材 token 降幅很大，packing 后每个 row 的可容纳内容会增加。
- 长度靠近边界的语文样本有 5 条从超限变为不超限。
- 当前仍超限的 5 条样本可以作为后续分块策略或提高窗口的重点观察对象。

不过，本实验只统计 tokenizer 输出长度，不测 GPU 吞吐。若要确认 DS tokenizer 对训练速度和显存的实际影响，需要在 DS 分支重新跑 sequence packing GPU benchmark。

## 7. 产物记录

本次实验控制台摘要：

```text
total_records     416
total_glm_tokens  2357420
total_ds_tokens   2226598
```

完整逐样本输出由以下命令生成：

```powershell
$paths = Get-ChildItem -LiteralPath .\data\structured -Filter '*.text.jsonl' |
  Where-Object { $_.Name -match '^(?:[1-9]|10|11)-+.*专升本考试专用教材' } |
  Sort-Object { [int]([regex]::Match($_.Name, '^\d+').Value) }

.\.venv\Scripts\python.exe tests\test_tokenizer_jsonl_token_counts.py @($paths.FullName)
```

## 8. 相关验证

本次实验前，`test_tokenizer_jsonl_token_counts.py` 已通过：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_tokenizer_jsonl_token_counts
.\.venv\Scripts\python.exe -m compileall tests\test_tokenizer_jsonl_token_counts.py
```

说明：

- 未设置 `TOKENIZER_COMPARE_JSONL` 时，该 unittest 会正常跳过。
- 直接传入一个或多个 JSONL 路径时，会打印逐样本统计。

## 9. 实验结论

扩大到 11 本专升本教材后，`ds_tokenizer` 相比 `glm_tokenizer` 的收益更明确：

- 总 token 数下降 `130822`
- 总体降幅约 `5.55%`
- `7680` 训练窗口下的超限样本从 `10` 条降到 `5` 条
- 最大 DS 正文长度为 `8770`

结论是：切换 DS tokenizer 对训练成本和截断风险都有正向收益。若继续使用 `7680`，大多数教材样本已经安全；若目标是完整覆盖这 11 本教材，建议后续评估 `8960` 训练窗口。
