# magdyn

虚构世界气候 / 地形工具集：从灰度海拔图生成温度图与等高线，并在浏览器中交互查看。

## 快速开始

**无参数**打开交互菜单；**有参数**则直接执行命令。

```bash
# Windows
magdyn.cmd

# Linux / macOS / Git Bash
./magdyn.sh
```

菜单项：`status` · `ensure` · `contours` · `temperature` · `summarize` · **`viewer`** · `pipeline` · **环境** · 切换语言

- 缺依赖时**不退出菜单**，显示报错，并提示在终端 `conda activate` 后重开；也可在「环境」里设置首选 conda 环境（写入 `.magdyn_env`，下次用 `conda run` 启动）。
- 气温菜单不再询问 downsample。

启动时可选 **中文 / English**（记住到 `.magdyn_lang`）。也可：

```bash
magdyn.cmd --lang zh
./magdyn.sh --lang en
python -m magdyn menu --lang zh
```

直接执行（跳过菜单）：

```bash
magdyn.cmd status
./magdyn.sh ensure
python -m magdyn viewer --port 8765
python -m magdyn temperature -- --cpu
```

也可用 `viewer.bat` 直接启动查看器。

## 数据约定

### 全海拔 `graphs/Terrain - Full Elevation.png`

单通道灰度，`0.5`（约 gray 128）= 海平面：

| 区间 | 含义 |
|------|------|
| `> 0.5` | 陆地；线性映射到 `0 … max_elev_m`（默认 **8000 m**） |
| `< 0.5` | 海洋；线性映射到 `0 … min_elev_m`（默认 **-8000 m**） |

海陆优先用 `Terrain - Land Mask.png`（白 = 陆）。缺失时 `ensure` / viewer 从全海拔推导（`> 0.5` = 陆）。

### 气温 `graphs/temperature/*.png`

```text
gray = clip((T_C − T_MIN) / (T_MAX − T_MIN) × 255)
```

默认 `T_MIN = −60 °C`，`T_MAX = +45 °C`。另有 `temperature_meta.json`、`temperature_stats.json`。

## 推荐流水线

```text
ensure → contours → temperature → summarize
```

或：

```bash
./magdyn.sh pipeline --temperature
```

| 步骤 | 说明 |
|------|------|
| `status` | 列出资源是否存在，并写 `graphs/assets.json` |
| `ensure` | 仅有全海拔时生成 Land Mask、Above/Below；可从 `graphs/template/` 播种 |
| `contours` | 陆地等高线（次级 200 m，主等高线 1000 m） |
| `temperature` | 12 月 + 年均气温（PyTorch，优先 CUDA；可用 `--cpu`、`--downsample`） |
| `summarize` | 打印摘要并写 `temperature_stats.json` |
| `viewer` | 本地 HTTP + 打开查看器（启动前自动 `ensure`） |
| `pipeline` | `ensure` → `contours`；可选 `--temperature` / `--force` |

气温常用参数见 `python -m magdyn temperature -- --help`（如 `--coast-blend-km`、`--heat-transport`、`--ocean-sst-nudge`）。

## 目录

```text
magdyn/
├── magdyn.cmd / magdyn.sh   # 交互菜单 / CLI 入口
├── viewer.bat               # 快捷打开 viewer
├── magdyn/                  # Python 包
│   ├── cli.py
│   ├── assets.py
│   ├── contours.py
│   ├── temperature.py
│   └── summarize.py
├── viewer/index.html
└── graphs/
    ├── assets.json
    ├── template/            # 可选：种子 Full Elevation
    ├── Terrain - *.png
    ├── Satellite Color.png  # 可选
    └── temperature/
```

兼容旧脚本名（`generate_temperature.py` 等）为指向包内模块的薄封装。

本地可选：`magdyn/reshape.py`（海拔非线性重映射，**已 gitignore**，不出现在交互菜单；有文件时可 `python -m magdyn reshape`）。

## Viewer

```bash
./magdyn.sh viewer
# → http://127.0.0.1:8765/viewer/
```

- **图层**：无卫星图则不显示；气温层按实际文件加载
- **图例**：常显；色阶 / 陆地描边 / 等高线 / **经纬度** 可切换
- **经纬网**：30° 网格；赤道、回归线（±23.5°）、极圈（±66.5°）加粗；本初子午线略强调
- **读数**：悬停与钉点（经纬、海陆、海拔/水深、气温）；钉点可看全年气温曲线
- **资源**：优先读 `graphs/assets.json`，否则逐文件探测

海拔色阶：≤0 浅蓝，>0 至少浅绿，约 1 / 1.5 / 2 / 3 km 处分色。解码与 README 一致：陆地 `0…+max_elev_m`，海洋 `0…min_elev_m`（默认 ±8000 m，线性）。

## 依赖

- Python 3.10+
- `numpy`、`Pillow`、`scipy`
- 气温：`torch`（有 CUDA 更快；`--cpu` 可强制 CPU）

## 备注

- 改地形后请重跑 `temperature`，否则气温层与新海拔可能不一致。
- `.gitignore` 可能忽略大图与温度产物；可将种子全海拔放在 `graphs/template/` 后执行 `ensure`。
