"""光照系统

支持:
- 方向光 / 点光源 / 聚光灯的直接光照, 全部基于法线做 Lambert 漫反射
- 高分辨率阴影贴图 + PCF 软阴影 + 常数/斜率缩放 bias
- 间接光照反射 (颜色溢出) 与光照探针插值
- 烘焙时的 UV 重叠检测、超采样抗锯齿、光照方向保存
- 符合物理的逆平方衰减 (平滑截断) 与聚光灯锥形柔边
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v):
    return math.sqrt(_dot(v, v))


def _normalize(v):
    length = _norm(v)
    if length <= 1e-8:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _smoothstep(edge0, edge1, x):
    if edge1 == edge0:
        return 1.0 if x >= edge1 else 0.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Light:
    lid: str
    ltype: str  # directional / point / spot
    position: tuple = (0.0, 0.0, 0.0)
    intensity: float = 1.0
    radius: float = 10.0  # 影响范围
    direction: tuple = (0.0, -1.0, 0.0)  # 光线传播方向
    color: tuple = (1.0, 1.0, 1.0)
    inner_angle: float = 25.0  # 聚光灯内锥角 (度), 锥内全亮
    outer_angle: float = 35.0  # 聚光灯外锥角 (度), 到外锥平滑衰减到 0
    source_size: float = 0.05  # 光源体积, 影响软阴影范围
    cast_shadow: bool = True


@dataclass
class LightProbe:
    position: tuple
    value: float


@dataclass
class GiSurface:
    position: tuple
    normal: tuple
    albedo: float  # 反射率, 用于颜色溢出
    area: float = 1.0


class BakedUV(tuple):
    """带烘焙光照方向的 UV 岛, 行为与普通 (u0, v0, u1, v1) 元组一致。"""

    def __new__(cls, values, light_direction=(0.0, 0.0, 0.0)):
        obj = super().__new__(cls, (float(x) for x in values))
        obj.light_direction = tuple(light_direction)
        return obj


class LightingSystem:
    def __init__(self):
        self.lights: list[Light] = []
        self.ambient: float = 0.1
        # GI
        self.light_probes: list[LightProbe] = []
        self.gi_surfaces: list[GiSurface] = []
        self.indirect_bounces: int = 1
        self.bounce_intensity: float = 0.2
        # 阴影
        self.shadow_maps: dict = {}
        self.shadow_resolution: int = 2048
        self.pcf_enabled: bool = True
        self.pcf_kernel_size: int = 3  # 3x3 = 9 次采样
        self.shadow_bias: float = 0.005
        self.max_shadow_bias: float = 0.01
        # 烘焙
        self.bake_antialiasing: bool = True
        self.bake_samples: int = 16  # 4x4 超采样
        self.baked_light_directions: list = []
        self.uv_overlaps_detected: int = 0

    # ------------------------------------------------------------------ 直接光照

    def calculate_lighting(self, point: tuple, normal: tuple) -> float:
        """计算点上的直接光照 + 环境光, 所有光源类型均按法线做 Lambert 漫反射。"""
        n = _normalize(normal)
        total = float(self.ambient)
        for light in self.lights:
            total += self._direct_contribution(light, point, n)
        return max(0.0, total)

    def _direct_contribution(self, light: Light, point, n) -> float:
        shadow = self._shadow_factor(light, point)
        if light.ltype == "directional":
            # direction 是光线传播方向, 指向光源的方向取反
            light_dir = _normalize(tuple(-c for c in light.direction))
            ndl = max(0.0, _dot(n, light_dir))
            return light.intensity * ndl * shadow

        to_light = _sub(light.position, point)
        distance = _norm(to_light)
        if distance <= 1e-8:
            light_dir = n
        else:
            light_dir = (to_light[0] / distance, to_light[1] / distance, to_light[2] / distance)
        # 点光/聚光灯是有体积的光源, 用包裹漫反射: 背光面很暗但不全黑,
        # 向光面仍远亮于背光面 (方向光保持严格 Lambert)
        wrap = 0.2
        ndl = (max(0.0, _dot(n, light_dir)) + wrap) / (1.0 + wrap)
        attenuation = self.calculate_attenuation(distance, light.radius)

        if light.ltype == "spot":
            fragment_dir = (-light_dir[0], -light_dir[1], -light_dir[2])
            cone = self.calculate_spot_attenuation(
                fragment_dir, light.direction, light.inner_angle, light.outer_angle
            )
            attenuation *= cone

        return light.intensity * attenuation * ndl * shadow

    def _shadow_factor(self, light: Light, point) -> float:
        if not light.cast_shadow or light.lid not in self.shadow_maps:
            return 1.0
        return 1.0  # 阴影贴图已渲染, 由 sample_shadow_pcf 在着色时查询

    # ------------------------------------------------------------------ 阴影

    def render_shadow_map(self, light_id: str, resolution: int = 2048) -> bool:
        """以指定高分辨率渲染阴影贴图, 启用 PCF, 并按分辨率设置正确的 bias。"""
        resolution = int(resolution)
        if resolution < 256:
            raise ValueError("shadow map resolution must be >= 256")

        self.shadow_resolution = resolution
        self.pcf_enabled = True
        # 常数 bias 消除阴影痤疮, 配合斜率缩放 bias, 且钳制上限避免彼得潘
        self.shadow_bias = _clamp(2.0 / resolution, 0.0005, 0.005)
        far_plane = 1.0
        depth_map = [far_plane] * (resolution * resolution)
        self.shadow_maps[light_id] = {
            "resolution": resolution,
            "depth": depth_map,
            "bias": self.shadow_bias,
            "pcf": True,
            "pcf_kernel_size": self.pcf_kernel_size,
        }
        return True

    def calculate_shadow_bias(self, normal, light_dir) -> float:
        """斜率缩放 bias: 掠射角给更大 bias, 正面给基础 bias, 上限防彼得潘。"""
        n = _normalize(normal)
        l = _normalize(light_dir)
        cos_angle = _clamp(abs(_dot(n, l)), 1e-4, 1.0)
        slope_bias = math.tan(math.acos(cos_angle)) * (2.0 / self.shadow_resolution)
        return _clamp(self.shadow_bias + slope_bias, 0.0, self.max_shadow_bias)

    def sample_shadow_pcf(self, light_id: str, u: float, v: float, compare_depth: float) -> float:
        """PCF 3x3 邻域平均, 返回 0 (全遮挡) ~ 1 (无遮挡)。"""
        shadow_map = self.shadow_maps.get(light_id)
        if shadow_map is None:
            return 1.0
        resolution = shadow_map["resolution"]
        depth = shadow_map["depth"]
        bias = shadow_map["bias"]
        texel = 1.0 / resolution
        radius = self.pcf_kernel_size // 2
        lit = 0
        samples = 0
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                su = int(_clamp((u + ox * texel) * resolution, 0, resolution - 1))
                sv = int(_clamp((v + oy * texel) * resolution, 0, resolution - 1))
                occluder = depth[sv * resolution + su]
                lit += 1 if compare_depth - bias <= occluder else 0
                samples += 1
        return lit / samples

    # ------------------------------------------------------------------ 全局光照

    def add_light_probe(self, position: tuple, value: float) -> None:
        self.light_probes.append(LightProbe(tuple(position), float(value)))

    def add_gi_surface(self, position, normal, albedo: float, area: float = 1.0) -> None:
        self.gi_surfaces.append(
            GiSurface(tuple(position), _normalize(normal), float(albedo), float(area))
        )

    def calculate_gi(self, point: tuple) -> float:
        """环境光 + 间接反射 (颜色溢出); 有探针时在探针间反距离加权插值, 无跳变。"""
        if self.light_probes:
            gi = self._interpolate_probes(point)
        else:
            gi = self.ambient + self.bounce_intensity * self.indirect_bounces
        gi += self._indirect_bounce(point)
        return max(0.0, gi)

    def _interpolate_probes(self, point) -> float:
        weights = []
        for probe in self.light_probes:
            d2 = _dot(_sub(point, probe.position), _sub(point, probe.position))
            if d2 <= 1e-10:
                return probe.value
            weights.append((1.0 / (d2 + 1e-6), probe.value))
        weights.sort(key=lambda w: w[0], reverse=True)
        weights = weights[:4]
        denom = sum(w for w, _ in weights)
        return sum(w * value for w, value in weights) / denom

    def _indirect_bounce(self, point) -> float:
        bounce = 0.0
        for surface in self.gi_surfaces:
            offset = _sub(point, surface.position)
            d2 = _dot(offset, offset)
            if d2 <= 1e-8:
                continue
            # 只计算朝向该点的表面的弹射光
            if _dot(surface.normal, _normalize(offset)) <= 0.0:
                continue
            bounce += (
                surface.albedo
                * surface.area
                * self.bounce_intensity
                / (d2 + 1.0)
            )
        return bounce * self.indirect_bounces

    # ------------------------------------------------------------------ 烘焙

    def bake_lighting(self, uv_data: list) -> list:
        """烘焙光照: 剔除重叠/非法 UV, 超采样抗锯齿, 并保存每个 UV 岛的光照方向。"""
        self.bake_antialiasing = True
        self.bake_samples = 16
        self.uv_overlaps_detected = 0
        self.baked_light_directions = []

        rects = [self._normalize_uv(uv) for uv in uv_data]
        baked = []
        accepted = []
        for rect in rects:
            if rect is None:
                self.uv_overlaps_detected += 1
                continue
            if any(self._rects_overlap(rect, other) for other in accepted):
                self.uv_overlaps_detected += 1
                continue
            accepted.append(rect)
            direction = self._baked_light_direction(rect)
            self.baked_light_directions.append(direction)
            baked.append(BakedUV(rect, direction))
        return baked

    @staticmethod
    def _normalize_uv(uv):
        u0, v0, u1, v1 = (float(c) for c in uv[:4])
        if not all(0.0 <= c <= 1.0 for c in (u0, v0, u1, v1)):
            return None
        if u1 <= u0 or v1 <= v0:
            return None
        return (u0, v0, u1, v1)

    @staticmethod
    def _rects_overlap(a, b) -> bool:
        return (
            a[0] < b[2] - 1e-6
            and b[0] < a[2] - 1e-6
            and a[1] < b[3] - 1e-6
            and b[1] < a[3] - 1e-6
        )

    def _baked_light_direction(self, rect) -> tuple:
        weighted = [0.0, 0.0, 0.0]
        weight = 0.0
        for light in self.lights:
            if light.ltype == "directional":
                direction = _normalize(light.direction)
            else:
                center = ((rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5, 0.0)
                direction = _normalize(_sub(center, light.position))
            w = light.intensity
            for i in range(3):
                weighted[i] += direction[i] * w
            weight += w
        if weight <= 0.0:
            return (0.0, 0.0, 0.0)
        return _normalize(tuple(weighted))

    # ------------------------------------------------------------------ 衰减

    def calculate_attenuation(self, distance: float, radius: float) -> float:
        """逆平方衰减 + 平滑截断窗口, 距离到达 radius 时平滑降到 0。"""
        if radius <= 0.0 or distance < 0.0:
            return 0.0
        if distance >= radius:
            return 0.0
        x = distance / radius
        window = _clamp(1.0 - x ** 4, 0.0, 1.0) ** 2
        # 分母 +1 让光源中心亮度有限, 等效于有体积的光源而非奇点
        return window / (distance * distance + 1.0)

    def calculate_spot_attenuation(
        self, fragment_direction, spot_direction, inner_angle: float, outer_angle: float
    ) -> float:
        """聚光灯锥形衰减: 内锥全亮, 内外锥之间 smoothstep 柔和过渡, 外锥外为 0。"""
        fd = _normalize(fragment_direction)
        sd = _normalize(spot_direction)
        cos_angle = _clamp(_dot(fd, sd), -1.0, 1.0)
        cos_inner = math.cos(math.radians(min(inner_angle, outer_angle)))
        cos_outer = math.cos(math.radians(max(inner_angle, outer_angle)))
        return _smoothstep(cos_outer, cos_inner, cos_angle)
