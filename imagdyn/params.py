"""IMagDyn global tunables — edit this file, then re-run temperature / wind.

IMagDyn 全局调参文件：改数字后重新跑气温 / 风场即可。

CLI flags still override these defaults. Planet fields are shared by
temperature, wind, currents, contours, and summarize.
命令行参数仍可覆盖此处默认值。行星参数由气温、风场、洋流、等高线与 summarize 共用。

Usage::

    from imagdyn.params import PLANET, TEMPERATURE, CURRENTS, WIND, ENCODE

    PLANET.radius_km          # 6371
    TEMPERATURE.greenhouse_factor
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# Numerical floor for 1/cos(φ) kernel stretch (~89.14°).
# 1/cos(φ) 核拉伸的数值下限（约 89.14°）。
METRIC_COS_EPS = 0.015

# Stefan–Boltzmann constant (W m^-2 K^-4); not a climate knob.
# 斯特藩–玻尔兹曼常数（W m^-2 K^-4）；一般不调。
STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass(frozen=True)
class PlanetParams:
    """Shared planetary geometry, radiation, and elevation scale.
    共用的行星几何、辐射与海拔尺度。
    """

    # Equatorial radius (km); sets km/px on the equirectangular grid.
    # 赤道半径（km）；决定等距圆柱网格上的 km/px。
    radius_km: float = 6371.0

    # TOA solar constant (W/m^2).
    # 大气顶太阳常数（W/m^2）。
    s0: float = 1361.0

    # Axial tilt (deg); tropics sit at ±this latitude.
    # 黄赤交角（度）；热带边界约为 ±该纬度。
    obliquity_deg: float = 23.5

    # Full-elevation gray 1.0 → this many metres ASL (ocean 0.0 → −max).
    # 高程图灰度 1.0 → 陆地海拔（m）；海洋灰度 0.0 → −max。
    max_elev_m: float = 8000.0

    # Environmental lapse on land (K/km); wind undoes this for SLP.
    # 陆地环境递减率（K/km）；风场计算海平面气压时会先撤销此项。
    lapse_k_per_km: float = 6.5

    # Sidereal / solar day length (hours) → Ω for Coriolis.
    # 自转周期（小时）→ 科里奥利参数 Ω。
    t_spin_hours: float = 24.0


@dataclass(frozen=True)
class TemperatureParams:
    """Temperature synthesizer knobs (see ``synthesize_temperatures``).
    气温合成参数（见 ``synthesize_temperatures``）。
    """

    # E-folding distance of sea-level temperature diffusion (km).
    # 海平面气温扩散的 e 折距离（km）。
    maritime_e_fold_km: float = 250.0

    # Greenhouse G: OLR = σT^4 / G (shortwave is not multiplied).
    # 温室因子 G：出射长波 OLR = σT^4 / G（短波不乘 G）。
    greenhouse_factor: float = 1.52

    # Newtonian heat transport λ (W/m²/K): Q_transport = λ (T̄_global − T_local).
    # 牛顿热输送 λ（W/m²/K）：Q_transport = λ (T̄_global − T_local)。
    # T̄_global is the area-weighted (dA ∝ cos φ) mean; the flux is energy-conserving.
    # T̄_global 为面积加权平均（dA ∝ cos φ）；该项全球积分为 0。
    transport_lambda: float = 1.8

    # Broadband albedo of open ocean.
    # 开阔洋面宽波段反照率。
    albedo_ocean: float = 0.11

    # Broadband albedo of ice-free land.
    # 无冰陆地宽波段反照率。
    albedo_land: float = 0.18

    # Broadband albedo of ice / snow.
    # 冰雪宽波段反照率。
    albedo_ice: float = 0.60

    # Repeated GPU mean-filter passes for sea-level T diffusion.
    # 海平面气温 GPU 均值滤波次数。
    maritime_diffuse_passes: int = 6

    # Land effective heat capacity (J/m²/K); ~2 m water equivalent.
    # 陆地有效热容（J/m²/K）；约 4 m 水当量。
    heat_capacity_land: float = 1.5e7

    # Deep-ocean reference heat capacity (J/m²/K); ~100 m water equivalent.
    # 深海参考热容（J/m²/K）；约 100 m 水当量。
    heat_capacity_ocean: float = 4.0e8

    # Depth-inertia scale at z = 0 (shelf / shallow).
    # z = 0（陆架 / 浅水）的深度惯性系数。
    inertia_shallow: float = 0.25

    # Depth-inertia scale as z → ∞ (deep ocean).
    # z → ∞（深海）的深度惯性系数。
    inertia_deep: float = 1.0

    # Characteristic mixed-layer depth d0 in I(z) = I_s + (I_d-I_s)(1-e^{-z/d0}) (m).
    # 深度惯性公式中的特征混合层深度 d0（m）。
    mix_depth_m: float = 200.0

    # Inland-lake heat-capacity coefficient at/below ``lake_small_area_km2``; max 1.
    # 面积 ≤ ``lake_small_area_km2`` 的内陆湖热容系数；最大为 1。
    lake_inertia: float = 0.45

    # Inland-lake area (km²) at which the coefficient = ``lake_inertia``.
    # 内陆湖系数取 ``lake_inertia`` 的面积阈值（km²）。
    lake_small_area_km2: float = 20_000.0

    # Max inland-lake area (km²); coefficient ramps to 1.0 here.
    # 内陆湖最大面积（km²）；系数在此升至 1.0。
    lake_max_area_km2: float = 50_000.0

    # Land / inland-lake freeze point (°C).
    # 陆地与内陆湖结冰点（°C）。
    freeze_land_C: float = 0.0

    # World-ocean freeze point (°C).
    # 世界大洋结冰点（°C）。
    freeze_ocean_C: float = -1.8

    # Gaussian width δT of latent-heat virtual capacity (°C).
    # 潜热虚拟热容高斯宽度 δT（°C）。
    freeze_latent_delta_C: float = 0.8

    # Equivalent ice thickness (m) setting the latent-heat capacity peak.
    # 决定潜热热容峰值的等效冰厚（m）。
    latent_ice_m: float = 0.5

    # Sigmoid half-width (°C) of ice-albedo transition.
    # 冰反照率 sigmoid 过渡半宽（°C）。
    ice_albedo_soft_C: float = 1.5

    # Virtual years of month-steps before keeping the last 12 months.
    # 保留最后 12 个月之前的虚拟自旋年数。
    spinup_years: int = 6

    # Uniform initial temperature (K).
    # 全图初值气温（K）。
    t_init_K: float = 280.0

    # Equatorial pixel width of coast anti-alias blend (EW stretched by 1/cos φ).
    # 海岸抗锯齿混合的赤道像素宽度（东西向按 1/cos φ 拉伸）。
    aa_blend_px: int = 1

    # Fold east-warm / west-cold current ΔT into Q_abs as an energy flux.
    # 是否把东暖西冷洋流 ΔT 作为能量通量加入 Q_abs。
    currents: bool = True

    # After temperature, also synthesize wind/pressure (CLI default).
    # 气温算完后是否接着合成风场 / 气压（CLI 默认）。
    sync_wind: bool = False


@dataclass(frozen=True)
class CurrentParams:
    """Ocean-current coastline filter (east-warm / west-cold).
    洋流海岸滤波（东岸暖 / 西岸冷）。
    """

    # |latitude| (deg) where east–west current contrast peaks.
    # 东暖西冷对比最强的 |纬度|（度）。
    peak_lat_deg: float = 30.0

    # Gaussian σ (° latitude) of the mid-latitude current weight.
    # 中纬洋流权重的高斯 σ（纬度°）。
    lat_sigma_deg: float = 12.0

    # East-coast (western-boundary) warm ΔT (°C).
    # 大陆东岸（西边界流）增温 ΔT（°C）。
    warm_delta_C: float = 3.0

    # West-coast (eastern-boundary) cold ΔT (°C).
    # 大陆西岸（东边界流）降温 ΔT（°C）。
    cold_delta_C: float = -3.0

    # Current footprint diffusion radius (km).
    # 洋流足迹扩散半径（km）。
    reach_km: float = 450.0

    # Diffusion passes for the standalone ``currents`` command.
    # 独立 ``currents`` 命令的扩散次数。
    diffuse_passes: int = 5

    # How far the ΔT bleeds onto land (standalone command; temperature path uses 0).
    # ΔT 向陆地渗入的比例（独立命令用；气温路径固定为 0）。
    land_bleed: float = 0.25


@dataclass(frozen=True)
class WindParams:
    """Wind / SLP synthesizer knobs (see ``WindField``).
    风场 / 海平面气压合成参数（见 ``WindField``）。
    """

    # Background mean sea-level pressure (hPa).
    # 背景平均海平面气压（hPa）。
    p0_hpa: float = 1013.25

    # Local thermal pressure conversion (hPa per °C); hot → low.
    # 局地热力气压换算（hPa/°C）；热低压。
    k_hpa_per_c: float = 1.2

    # Planetary-belt amplitude as a fraction of ``k_hpa_per_c``.
    # 行星气压带振幅，相对 ``k_hpa_per_c`` 的比例。
    belt_k_frac: float = 0.22

    # Cap on planetary-belt anomaly amplitude (hPa).
    # 行星气压带异常振幅上限（hPa）。
    belt_amp_max_hpa: float = 7.0

    # Amplitude of secondary (Ferrel-side) belt anomalies vs primary.
    # 次级（费雷尔一侧）气压带异常相对主带的比例。
    secondary_frac: float = 0.35

    # Latitude smoothing width (deg) after cosine belt segments.
    # 余弦气压带拼接后的纬度平滑宽度（度）。
    belt_blend_deg: float = 1.5

    # Base Rayleigh-drag coefficient for the AMC path integral (1/s).
    # 角动量路径积分用的瑞利摩擦基数（1/s）。
    drag_kappa0: float = 1.4e-6

    # Latitude factor on Rayleigh drag (stronger toward the equator).
    # 瑞利摩擦的纬度因子（赤道附近更强）。
    drag_kappa_lat: float = 3.0

    # Hadley-cell upper-branch meridional speed (m/s) for AMC.
    # 哈德利环流高空支的经向风速（m/s），用于角动量积分。
    hadley_v_m_s: float = 2.8

    # Tropopause height (m) in the thermal-wind / AMC scaling.
    # 热成风 / 角动量积分中的对流层顶高度（m）。
    tropopause_h_m: float = 10_000.0

    # Thermal-wind scale (m) converting ΔT to critical zonal wind.
    # 把 ΔT 换成临界纬向风的热成风尺度（m）。
    thermal_wind_scale: float = 6500.0

    # Floor on critical zonal wind ``u_crit`` (m/s) for subtropical highs.
    # 副高判定用临界纬向风 ``u_crit`` 下限（m/s）。
    u_crit_min_m_s: float = 20.0

    # Cap on critical zonal wind ``u_crit`` (m/s).
    # 临界纬向风 ``u_crit`` 上限（m/s）。
    u_crit_max_m_s: float = 70.0

    # Max |latitude| (deg) for the AMC path integral.
    # 角动量路径积分允许的最大 |纬度|（度）。
    amc_max_lat_abs: float = 50.0

    # Number of longitude sectors for planetary belts.
    # 行星气压带的经度扇区数。
    belt_lon_sectors: int = 36

    # Linear drag leftover used in some force scalings (1/s).
    # 部分力平衡里仍用到的线性阻力（1/s）。
    drag: float = 1.2e-5

    # Quadratic surface-drag coeff c_d (1/m) over ocean: F = −c_d |V| V.
    # 洋面二次方地表阻力系数 c_d（1/m）：F = −c_d |V| V。
    friction_ocean: float = 3.0e-6

    # Quadratic surface-drag coeff c_d (1/m) over land.
    # 陆地二次方地表阻力系数 c_d（1/m）。
    friction_land: float = 1.0e-5

    # Air density (kg/m^3) in the pressure-gradient / drag balance.
    # 气压梯度与阻力平衡中的空气密度（kg/m^3）。
    air_density: float = 1.2

    # Hard cap on wind speed (m/s).
    # 风速硬上限（m/s）。
    speed_cap: float = 60.0

    # Fractional slowdown of upslope flow.
    # 迎风坡减速比例。
    upslope_slow: float = 0.70

    # Fractional speed-up of downslope (lee) flow.
    # 背风坡加速比例。
    downslope_boost: float = 0.20

    # Near-zero speed (m/s) treated as blocked by terrain.
    # 被地形挡住时视作近零的风速（m/s）。
    block_speed: float = 3.0

    # Fraction of blocked flow diverted along-slope.
    # 被挡住气流沿坡分流的比例。
    divert_frac: float = 0.28

    # Slope (m/km) above which terrain blocking kicks in.
    # 开始触发地形阻挡的坡度（m/km）。
    slope_block_m_per_km: float = 35.0

    # Overall scale of the pressure-gradient force.
    # 气压梯度力的总缩放。
    force_scale: float = 1.0

    # |lat| ≥ this (deg) is forced to ocean with elevation → 0.
    # |纬度| ≥ 此值（度）强制为海洋，海拔归零。
    polar_ocean_lat: float = 89.0

    # |lat| where east–west pressure anomaly starts fading to zonal mean (deg).
    # 东西向气压异常开始向纬向平均消退的 |纬度|（度）。
    polar_fade_lat: float = 87.0

    # Gaussian σ (px) for softening the land/ocean coast in terrain prep.
    # 地形预处理里柔化海陆边界的高斯 σ（像素）。
    coast_sigma_px: float = 6.0

    # Gaussian σ (px) of the ∇p convolution.
    # 气压梯度卷积的高斯 σ（像素）。
    grad_sigma_px: float = 3.0

    # Gaussian σ (px) for neighborhood wind-direction smoothing.
    # 邻域风向平滑的高斯 σ（像素）。
    wind_smooth_sigma_px: float = 1.0

    # Land keeps this fraction of a fully blocked high-pressure belt anomaly.
    # 陆地在高压带被完全挡住时仍保留的异常比例。
    land_belt_frac: float = 0.2

    # Sigmoid half-width (hPa) of the land belt-block gate.
    # 陆地气压带阻挡门控的 sigmoid 半宽（hPa）。
    land_block_half_hpa: float = 1.5

    # Gaussian σ (deg) of the equatorial notch on local thermal → pressure.
    # 局地热力→气压在赤道附近减弱的高斯 σ（度）。
    thermal_lat_sigma_deg: float = 15.0

    # Residual thermal→pressure weight exactly at the equator [0, 1].
    # 赤道上局地热力→气压的残留权重 [0, 1]。
    thermal_equator_frac: float = 0.1

    # Mild 2D diffusion σ (px) of total SLP after belts + thermal.
    # 气压带与热力异常叠加后，总 SLP 的轻度 2D 扩散 σ（像素）。
    pressure_smooth_sigma_px: float = 8.0


@dataclass(frozen=True)
class EncodeParams:
    """PNG gray encoding for temperature maps.
    气温图的 PNG 灰度编码。
    """

    # Temperature (°C) mapped to gray 0.
    # 映射为灰度 0 的气温（°C）。
    t_gray_min: float = -60.0

    # Temperature (°C) mapped to gray 255.
    # 映射为灰度 255 的气温（°C）。
    t_gray_max: float = 45.0


@dataclass(frozen=True)
class ContourParams:
    """Land contour intervals (metres) for ``contours``.
    ``contours`` 命令的陆地等高线间距（米）。
    """

    # Minor contour interval (m).
    # 次等高线间距（m）。
    minor_m: float = 200.0

    # Major contour interval (m); drawn thicker.
    # 主等高线间距（m）；绘制更粗。
    major_m: float = 1000.0


# Frozen snapshots used by temperature / wind / currents / CLI defaults.
# 气温 / 风场 / 洋流 / CLI 默认值使用的冻结快照。
PLANET = PlanetParams()

TEMPERATURE = TemperatureParams()

CURRENTS = CurrentParams()

WIND = WindParams()

ENCODE = EncodeParams()

CONTOURS = ContourParams()


def temperature_call_kwargs(**overrides: Any) -> dict[str, Any]:
    """Kwargs for ``synthesize_temperatures`` (everything except arrays / device).
    传给 ``synthesize_temperatures`` 的关键字参数（不含数组与 device）。
    """
    p, t, c = PLANET, TEMPERATURE, CURRENTS
    kw: dict[str, Any] = dict(
        s0=p.s0,
        obliquity_deg=p.obliquity_deg,
        max_elev_m=p.max_elev_m,
        lapse_k_per_km=p.lapse_k_per_km,
        planet_radius_km=p.radius_km,
        maritime_e_fold_km=t.maritime_e_fold_km,
        greenhouse_factor=t.greenhouse_factor,
        transport_lambda=t.transport_lambda,
        albedo_ocean=t.albedo_ocean,
        albedo_land=t.albedo_land,
        albedo_ice=t.albedo_ice,
        maritime_diffuse_passes=t.maritime_diffuse_passes,
        heat_capacity_land=t.heat_capacity_land,
        heat_capacity_ocean=t.heat_capacity_ocean,
        inertia_shallow=t.inertia_shallow,
        inertia_deep=t.inertia_deep,
        mix_depth_m=t.mix_depth_m,
        lake_inertia=t.lake_inertia,
        lake_small_area_km2=t.lake_small_area_km2,
        lake_max_area_km2=t.lake_max_area_km2,
        freeze_land_C=t.freeze_land_C,
        freeze_ocean_C=t.freeze_ocean_C,
        freeze_latent_delta_C=t.freeze_latent_delta_C,
        latent_ice_m=t.latent_ice_m,
        ice_albedo_soft_C=t.ice_albedo_soft_C,
        spinup_years=t.spinup_years,
        t_init_K=t.t_init_K,
        aa_blend_px=t.aa_blend_px,
        currents=t.currents,
        current_warm_delta_C=c.warm_delta_C,
        current_cold_delta_C=c.cold_delta_C,
        current_peak_lat_deg=c.peak_lat_deg,
        current_lat_sigma_deg=c.lat_sigma_deg,
        current_reach_km=c.reach_km,
    )
    kw.update(overrides)
    return kw


def wind_field_kwargs(**overrides: Any) -> dict[str, Any]:
    """Kwargs for ``WindField`` (includes planet spin / radius / lapse).
    传给 ``WindField`` 的关键字参数（含自转、半径、递减率）。
    """
    kw = asdict(WIND)
    kw["planet_radius_km"] = PLANET.radius_km
    kw["lapse_k_per_km"] = PLANET.lapse_k_per_km
    kw["t_spin_s"] = PLANET.t_spin_hours * 3600.0
    kw.update(overrides)
    return kw


def all_as_dict() -> dict[str, dict[str, Any]]:
    """Snapshot of every tunable group (e.g. for meta JSON).
    全部调参组的快照（例如写入 meta JSON）。
    """
    return {
        "planet": asdict(PLANET),
        "temperature": asdict(TEMPERATURE),
        "currents": asdict(CURRENTS),
        "wind": asdict(WIND),
        "encode": asdict(ENCODE),
        "contours": asdict(CONTOURS),
        "metric_cos_eps": METRIC_COS_EPS,
    }
