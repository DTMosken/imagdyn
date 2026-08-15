# 气温合成

*[English](temperature.en.md) | 中文*

模块：[`imagdyn/temperature.py`](../imagdyn/temperature.py)、洋流 [`imagdyn/currents.py`](../imagdyn/currents.py)。

## 流程概要

1. 由全海拔与陆掩膜得到陆地高度与软海岸。
2. 12 个月：TOA 日均辐射（轴向倾角默认 ±23.5°）→ 年均值 + 热量输送 → 辐射平衡气温。
3. 海事影响：以「开放洋面」为源做均值扩散，得到 `maritime`。
4. 大陆度 / 季节敏感度、极地海洋阻尼、海拔修正。
5. 海水温度向陆地海平面目标扩散；生成洋流 ΔT（东岸暖 / 西岸冷，纬度高斯约 30°）。
6. 海岸陆海混合；结冰阈值以下水体当月不提供开放水域影响。

## 湖泊惯性

连通水体中，面积 ≤ `--lake-max-area-km2`（默认 20 000 km²）的内陆湖，热惯性降低为 `--lake-inertia`（默认 **0.6**）；大洋使用 `--ocean-inertia`（默认 0.90）。

## 常用 CLI

```bash
python -m imagdyn temperature -- --help
python -m imagdyn temperature -- --cpu --no-wind --no-currents
python -m imagdyn currents -- --dump-maps
```

关键参数：`--greenhouse`、`--heat-transport`、`--maritime-efold-km`、`--maritime-diffuse-passes`、`--ocean-sst-nudge`、`--lake-inertia`、`--lake-max-area-km2`。

返回 [README](../README.md) · [数据格式](data-formats.md)。
