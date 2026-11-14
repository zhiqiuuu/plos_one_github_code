# 可复现性审计

本次运行读取仓库外的受限 TALIS 原始教师微观数据，仅在仓库中保留派生、非识别性结果。`B01/reproduction.log` 记录运行阶段；`B01/brr_replicate_estimates.csv` 保存 100 个复本的逐参数估计；`B01/brr_failures.csv` 保存失败记录。请先运行 `code/01_data_preparation.py`，再运行 `code/run_all.py`。
