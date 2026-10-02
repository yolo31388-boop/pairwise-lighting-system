"""光照系统 - 含5个bug"""
from dataclasses import dataclass, field

@dataclass
class Light:
    lid: str
    ltype: str  # directional/point/spot
    position: tuple
    intensity: float
    radius: float = 10.0

class LightingSystem:
    def __init__(self):
        self.lights: list[Light] = []
        self.ambient: float = 0.1

    def calculate_lighting(self, point: tuple, normal: tuple) -> float:
        # bug1: 只算方向光，不算法线
        total = self.ambient
        for light in self.lights:
            if light.ltype == "directional":
                total += light.intensity
        return total

    def render_shadow_map(self, light_id: str, resolution: int = 512) -> bool:
        # bug2: 固定分辨率，无PCF
        return True

    def calculate_gi(self, point: tuple) -> float:
        # bug3: 只算环境光
        return self.ambient

    def bake_lighting(self, uv_data: list) -> list:
        # bug4: 不检查UV重叠
        return uv_data

    def calculate_attenuation(self, distance: float, radius: float) -> float:
        # bug5: 线性衰减
        return max(0, 1 - distance / radius)
