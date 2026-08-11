# 风场与气压

模块：[`imagdyn/wind.py`](../imagdyn/wind.py)。

## 流程概要

1. 地表气温海平面化（撤销陆地 lapse）→ 驱动 SLP。
2. **行星气压带**（经度均分 **36** 扇区）：
   - 扇区内平均气温 → 热赤道；
   - 瑞利摩擦角动量路径积分 + 热成风 `u_crit` → 副高；
   - 副高极侧 `|dT̄/dφ|` 最大 → 副极地低压；
   - 极地高压固定 **±88°**；
   - 相邻气压带之间用**余弦半波**连接异常（约 1.5° 纬度 blend）；
   - 扇区结果经度**周期插值**复合；meta 记平均位置与 `upper_amc`。
3. 局地热力气压异常（赤道附近减弱）+ 陆地对高压异常的动态阻尼。
4. 2D 扩散（经度环绕）；**85°→88°** 渐消东西向异常。
5. ∇p → 科里奥利 + **二次方**地表阻力平衡；UV 邻域卷积。
6. 地形阻挡 / 分流 / 背风（水体海拔按 0；极地 `|lat|≥88` 强制海洋）。

默认地图自转周期 **24 h**（赤道）。

## 产物

见 [数据格式 · 风场](data-formats.md#风场-graphswind)。`summarize` / `wind` 会写 `wind_stats.json`（含全海洋、海拔 0 的 1D 水世界三组太阳直射情形）。

## 常用 CLI

```bash
python -m imagdyn wind --
python -m imagdyn wind -- --annual-only --cpu
python -m imagdyn temperature -- --no-wind
```

返回 [README](../README.md) · [数据格式](data-formats.md)。
