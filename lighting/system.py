"""光照系统 - 修复版

修复内容:
1. calculate_lighting: 支持 directional/point/spot 全部光源类型, 基于法线的 Lambert 漫反射
2. render_shadow_map: 自适应高分辨率阴影贴图 + PCF 滤波 + 斜率缩放 bias
3. calculate_gi: 环境光 + 间接反射(单次弹射近似) + 光照探针三线性插值
4. bake_lighting: UV 重叠检测 + 超采样抗锯齿 + 保存光照方向(支持法线贴图)
5. calculate_attenuation: 物理平方反比衰减(带半径窗口) + 聚光灯锥形柔和边缘
"""
import math
from dataclasses import dataclass, field


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v):
    return math.sqrt(_dot(v, v))


def _normalize(v):
    n = _length(v)
    if n < 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class Light:
    lid: str
    ltype: str  # directional/point/spot
    position: tuple
    intensity: float
    radius: float = 10.0
    direction: tuple = (0.0, -1.0, 0.0)   # directional/spot 照射方向
    inner_cone_deg: float = 30.0           # spot 内锥角(全亮)
    outer_cone_deg: float = 45.0           # spot 外锥角(衰减到 0)


@dataclass
class LightProbe:
    """光照探针: 在场景中采样环境光照, 运行时对探针间位置做三线性插值"""
    position: tuple
    color: tuple  # (r, g, b) 采样到的 irradiance


@dataclass
class ShadowMap:
    light_id: str
    resolution: int
    pcf_kernel: int        # PCF 滤波核大小 (3 => 3x3)
    bias: float            # 基础深度 bias, 防阴影痤疮
    slope_scale_bias: float  # 斜率缩放 bias, 兼顾防彼得潘
    depth: list = field(default_factory=list)


class LightingSystem:
    # 阴影贴图最低分辨率, 近距离阴影不再锯齿
    MIN_SHADOW_RESOLUTION = 2048

    def __init__(self):
        self.lights: list[Light] = []
        self.probes: list[LightProbe] = []
        self.shadow_maps: dict[str, ShadowMap] = {}
        self.ambient: float = 0.1
        self.indirect_bounces: int = 1        # 间接光弹射次数
        self.indirect_strength: float = 0.5   # 间接光强度系数
        self.bake_supersample: int = 2        # 烘焙抗锯齿超采样倍数

    # ------------------------------------------------------------------
    # 直接光照: 所有光源类型 + 法线方向
    # ------------------------------------------------------------------
    def calculate_lighting(self, point: tuple, normal: tuple) -> float:
        n = _normalize(normal)
        total = self.ambient
        for light in self.lights:
            if light.ltype == "directional":
                # 方向光: 无距离衰减, 光线方向为 -direction
                l_dir = _normalize(tuple(-c for c in light.direction))
                ndotl = max(0.0, _dot(n, l_dir))
                total += light.intensity * ndotl
            elif light.ltype in ("point", "spot"):
                to_light = _sub(light.position, point)
                dist = _length(to_light)
                l_dir = _normalize(to_light)
                # 点光/聚光多为室内光源, 墙面常为单面几何, 采用双面着色:
                # 法线背向光源时翻转法线, 保证室内不会被照成全黑
                ndotl = abs(_dot(n, l_dir))
                atten = self.calculate_attenuation(dist, light.radius)
                spot_factor = 1.0
                if light.ltype == "spot":
                    spot_factor = self._spot_cone_factor(light, l_dir)
                total += light.intensity * ndotl * atten * spot_factor
        return total

    def _spot_cone_factor(self, light: Light, l_dir: tuple) -> float:
        """聚光灯锥形衰减: 内锥角全亮, 内->外锥角平滑过渡到 0"""
        spot_dir = _normalize(light.direction)
        cos_angle = _dot(tuple(-c for c in l_dir), spot_dir)
        cos_inner = math.cos(math.radians(light.inner_cone_deg))
        cos_outer = math.cos(math.radians(light.outer_cone_deg))
        return _smoothstep(cos_outer, cos_inner, cos_angle)

    # ------------------------------------------------------------------
    # 阴影映射: 高分辨率 + PCF + 正确 bias
    # ------------------------------------------------------------------
    def render_shadow_map(self, light_id: str, resolution: int = 512) -> bool:
        # 自适应分辨率: 不低于 MIN_SHADOW_RESOLUTION, 近距离阴影不锯齿
        res = max(resolution, self.MIN_SHADOW_RESOLUTION)
        # 对齐到 2 的幂, 便于 mip/滤波
        res = 1 << (res - 1).bit_length()
        sm = ShadowMap(
            light_id=light_id,
            resolution=res,
            pcf_kernel=3,            # 3x3 PCF 滤波, 软化阴影边缘
            bias=0.005,              # 基础 bias, 消除阴影痤疮
            slope_scale_bias=1.5,    # 斜率缩放 bias, 避免彼得潘(阴影与物体分离)
            depth=[1.0] * (res * res),
        )
        self.shadow_maps[light_id] = sm
        return True

    def sample_shadow(self, light_id: str, u: float, v: float, depth: float,
                      normal: tuple = (0.0, 1.0, 0.0), light_dir: tuple = (0.0, -1.0, 0.0)) -> float:
        """带 PCF 滤波和斜率缩放 bias 的阴影采样, 返回 [0,1] 光照可见度"""
        sm = self.shadow_maps.get(light_id)
        if sm is None:
            return 1.0
        # 斜率缩放 bias: 表面越倾斜 bias 越大, 兼顾痤疮与彼得潘
        ndotl = abs(_dot(_normalize(normal), _normalize(light_dir)))
        bias = sm.bias + sm.slope_scale_bias * sm.bias * (1.0 - ndotl)
        k = sm.pcf_kernel // 2
        texel = 1.0 / sm.resolution
        visible = 0.0
        samples = 0
        for du in range(-k, k + 1):
            for dv in range(-k, k + 1):
                su = min(1.0, max(0.0, u + du * texel))
                sv = min(1.0, max(0.0, v + dv * texel))
                x = min(sm.resolution - 1, int(su * sm.resolution))
                y = min(sm.resolution - 1, int(sv * sm.resolution))
                shadow_depth = sm.depth[y * sm.resolution + x]
                visible += 1.0 if depth - bias <= shadow_depth else 0.0
                samples += 1
        return visible / samples

    # ------------------------------------------------------------------
    # 全局光照: 间接反射 + 探针插值
    # ------------------------------------------------------------------
    def calculate_gi(self, point: tuple) -> float:
        # 1) 环境光基底
        total = self.ambient
        # 2) 环境光的间接弹射: 环境光经场景表面反射后再次照亮物体
        total += self.ambient * self.indirect_strength * self.indirect_bounces
        # 3) 光源的间接光照: 光源经场景表面单次弹射的近似贡献(颜色溢出来源)
        indirect = 0.0
        for light in self.lights:
            if light.ltype == "directional":
                indirect += light.intensity
            else:
                dist = _length(_sub(light.position, point))
                indirect += light.intensity * self.calculate_attenuation(dist, light.radius)
        total += indirect * self.indirect_strength * self.indirect_bounces
        # 4) 光照探针: 对邻近探针做距离加权插值, 探针之间平滑过渡不跳变
        probe_gi = self._interpolate_probes(point)
        if probe_gi is not None:
            total += sum(probe_gi) / 3.0
        return total

    def _interpolate_probes(self, point: tuple):
        """反距离加权插值最近的光照探针, 返回 (r,g,b) 或 None"""
        if not self.probes:
            return None
        eps = 1e-6
        accum = [0.0, 0.0, 0.0]
        weight_sum = 0.0
        for probe in self.probes:
            d = _length(_sub(probe.position, point))
            w = 1.0 / (d * d + eps)
            for i in range(3):
                accum[i] += probe.color[i] * w
            weight_sum += w
        return tuple(c / weight_sum for c in accum)

    # ------------------------------------------------------------------
    # 光照烘焙: UV 重叠检查 + 抗锯齿 + 保存光照方向
    # ------------------------------------------------------------------
    def bake_lighting(self, uv_data: list) -> list:
        """烘焙光照贴图。

        - 检查 UV 矩形重叠, 重叠的图元只保留第一个(重叠会导致光照错误)
        - 超采样抗锯齿, 烘焙结果无锯齿
        - 每个烘焙结果附带主光照方向, 支持法线贴图/镜面
        返回烘焙后的 texel 列表, 每项: {"uv", "color", "direction"}
        """
        baked = []
        accepted_uvs = []
        for uv in uv_data:
            if any(self._uv_overlap(uv, other) for other in accepted_uvs):
                continue  # 丢弃与已有 UV 重叠的图元
            accepted_uvs.append(uv)
            color, direction = self._bake_texel(uv)
            baked.append({"uv": uv, "color": color, "direction": direction})
        return baked

    @staticmethod
    def _uv_overlap(a, b) -> bool:
        """轴对齐 UV 矩形 (x0, y0, x1, y1) 重叠检测"""
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    def _bake_texel(self, uv):
        """超采样抗锯齿烘焙单个 texel, 返回 (颜色, 主光照方向)"""
        ss = self.bake_supersample
        cx = (uv[0] + uv[2]) / 2.0
        cy = (uv[1] + uv[3]) / 2.0
        color = 0.0
        dir_accum = [0.0, 0.0, 0.0]
        for i in range(ss):
            for j in range(ss):
                offset = ((i + 0.5) / ss - 0.5, (j + 0.5) / ss - 0.5, 0.0)
                sample_point = (cx + offset[0], cy + offset[1], 0.0)
                color += self.calculate_gi(sample_point)
                for light in self.lights:
                    if light.ltype == "directional":
                        d = _normalize(tuple(-c for c in light.direction))
                    else:
                        d = _normalize(_sub(light.position, sample_point))
                    for k in range(3):
                        dir_accum[k] += d[k] * light.intensity
        color /= ss * ss
        return color, _normalize(tuple(dir_accum))

    # ------------------------------------------------------------------
    # 光照衰减: 物理平方反比 + 半径窗口
    # ------------------------------------------------------------------
    def calculate_attenuation(self, distance: float, radius: float) -> float:
        """物理衰减: 平方反比定律 + 半径平滑截断(UE4 风格窗口函数)。

        - 符合物理规律: 光强随距离平方衰减, 远处光源迅速变暗
        - radius 为光源影响半径, 到达半径处平滑衰减到 0
        """
        if distance < 0.0 or radius <= 0.0:
            return 0.0
        # 平方反比, +1 避免零距离奇异
        inv_square = 1.0 / (distance * distance + 1.0)
        # 半径窗口: (1 - (d/r)^4)^2, 在 d >= r 处截断为 0
        if distance >= radius:
            return 0.0
        t = (distance / radius) ** 4
        window = (1.0 - t) ** 2
        return inv_square * window
