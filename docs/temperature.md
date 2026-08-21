# 气温合成

*[English](temperature.en.md) | 中文*

模块：[`imagdyn/temperature.py`](../imagdyn/temperature.py)、洋流 [`imagdyn/currents.py`](../imagdyn/currents.py)。默认数值集中在 [`imagdyn/params.py`](../imagdyn/params.py)。

## 流程概要

1. 由全海拔与陆掩膜得到陆地高度、水体深度与软海岸。东西向核按 ``1/cos(φ)`` 拉伸，避免极地海岸变宽。
2. 全图初值（默认 280 K），GPU 上跑 ``spinup_years``（默认 **6**）个虚拟年。每个 month_step：冰反照率 sigmoid → ``C_eff = C_base + C_latent`` → ``Q_abs = (1-α)Q + Q_curr + Q_transport`` → 隐式能量更新 ``T_new = T + (Q_abs − σT⁴/G) / (C/Δt + 4σT³/G)`` → 折算海平面气温、度量扩散、再减 lapse。洋流 ΔT 在循环前算一次，每月换成 ``Q_curr = (4σT³/G) ΔT`` 只加在世界大洋。热输送 ``Q_transport = λ (T̄_global − T_local)``，``λ`` 为 ``transport_lambda``（默认 **3.8** W/m²/K），``T̄_global`` 为面积加权（``dA ∝ cos(φ)``）全球平均。
3. 只保留 **最后 12 个月** 写图与统计（预热不进 meta / stats）。
4. 海岸仅做 ``aa_blend_px``（默认 1 赤道像素）抗锯齿，无大半径陆海混合。

## 热容与结冰

- 陆地：``heat_capacity_land``。
- 世界洋：``heat_capacity_ocean · I(z)``，``I(z) = I_shallow + (I_deep − I_shallow)(1 − e^{−z/d0})``，``d0 = mix_depth_m``（默认 200 m）。
- 内陆湖：连通水体按 **纬度补正面积** ``dA ∝ cos(φ)``。面积 ≤ ``lake_max_area_km2`` 时系数为 ``lake_inertia``（默认 0.45）→ 在 ``lake_max_area_km2`` 线性升至 **1**。结冰点陆地/湖 **0.0 °C**，大洋 **−1.8 °C**。潜热为结冰点高斯虚拟热容（``δT`` 默认 0.8 °C）。

温室因子 ``G`` 只压长波（``OLR = σT⁴/G``），短波不乘 ``G``。

## 常用 CLI

```bash
python -m imagdyn temperature -- --help
python -m imagdyn temperature -- --cpu --no-wind --no-currents
python -m imagdyn currents -- --dump-maps
```

关键参数：`--greenhouse`、`--transport-lambda`、`--heat-capacity-land`、`--heat-capacity-ocean`、`--spinup-years`、`--maritime-efold-km`、`--lake-inertia`、`--freeze-land-c`、`--freeze-ocean-c`。

返回 [README](../README.md) · [数据格式](data-formats.md)。
