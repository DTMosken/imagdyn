# 数据格式

*[English](data-formats.en.md) | 中文*

IMagDyn 读写的主要栅格与 JSON 约定。

## 高程图 `graphs/Terrain - Full Elevation.png`

单通道灰度，**0.5**（约 gray 128）= 海平面：

| 区间 | 含义 |
|------|------|
| `> 0.5` | 陆地 → 线性映射到 `0 … max_elev_m`（默认 **+8000 m**） |
| `< 0.5` | 海洋 → 线性映射到 `0 … min_elev_m`（默认 **−8000 m**） |

海陆优先读 `Terrain - Land Mask.png`（白 = 陆）。没有遮罩时，`ensure` / viewer 用海拔 `> 0.5` 推导。

派生层（`ensure`）：

- `Terrain - Land Mask.png`
- `Terrain - Above Sea Level.png` / `Terrain - Below Sea Level.png`
- 可选 `Satellite Color.png`

示例种子：`graphs/template/`（现实地球全海拔），菜单 `2 ensure` 可播种到 `graphs/`。

## 气温 `graphs/temperature/*.png`

```text
gray = clip((T_C − T_MIN) / (T_MAX − T_MIN) × 255)
```

默认 `T_MIN = −60 °C`，`T_MAX = +45 °C`。同目录：

- `temperature_meta.json` — 生成参数与摘要
- `temperature_stats.json` — `summarize` 写出的分区统计

月文件名形如 `Temperature - 01 January.png`；年均 `Temperature - Annual Mean.png`。

## 风场 `graphs/wind/`

| 产物 | 格式 |
|------|------|
| `Wind - UV - *.png` | RGB8：U→R，V→G，气压→B |
| `Wind - Terrain Dot - *.png` | RGB：terrain_dot→R,G（16-bit 拆分），风速→B（8-bit） |
| `wind_meta.json` | 尺度、气压/风速图例、各期 belt / AMC 摘要 |
| `wind_stats.json` | 气压/风速分区统计 + 理想水世界 1D 剖面 |

物理约定（写入 meta）：

- 存储 UV：`+U` 向东，`+V` 向南（图像行方向）
- 气压为海平面气压（SLP）：陆地气温先按海拔订正后再做气压异常

## 资源清单

`graphs/assets.json` 由 `status` / `ensure` 维护，viewer 优先读取；缺失时再逐文件探测。

返回 [README](../README.md)。
