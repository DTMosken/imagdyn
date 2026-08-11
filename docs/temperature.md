# 气温合成

模块：[`imagdyn/temperature.py`](../imagdyn/temperature.py)、洋流 [`imagdyn/currents.py`](../imagdyn/currents.py)。

## 流程概要

1. 由全海拔与陆掩膜得到陆地高度（lapse）与软海岸。
2. 12 个月：TOA 日均辐射（轴向倾角默认 ±23.5°）→ 年均值 + 热量输送 → 辐射平衡气温。
3. 海事影响：以「开放洋面」为源做 GPU 可分离均值扩散（替代 EDT），得到 `maritime` 与伪距离。
4. 大陆度 / 季节敏感度、极地海洋阻尼、海拔 lapse。
5. 海水单元向 Earth-like SST 气候目标 nudge；可选洋流 ΔT（东岸暖 / 西岸冷，纬度高斯约 30°）。
6. 海岸陆海互拉与洋面混合；结冰阈值以下水体当月不提供开放洋影响。

## 湖泊惯性

连通水体中，**非最大洋盆**且面积 ≤ `--lake-max-area-km2`（默认 20 000 km²）的内陆湖，热惯性改为 `--lake-inertia`（默认 **0.6**）；大洋使用 `--ocean-inertia`（默认 0.90）。

## 常用 CLI

```bash
python -m imagdyn temperature -- --help
python -m imagdyn temperature -- --cpu --no-wind --no-currents
python -m imagdyn currents -- --dump-maps
```

关键参数：`--greenhouse`、`--heat-transport`、`--maritime-efold-km`、`--maritime-diffuse-passes`、`--ocean-sst-nudge`、`--lake-inertia`、`--lake-max-area-km2`。

返回 [README](../README.md) · [数据格式](data-formats.md)。
