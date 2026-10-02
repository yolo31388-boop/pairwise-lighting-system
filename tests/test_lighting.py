"""光照系统 - 红态测试"""
import pytest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lighting.system import LightingSystem, Light

class TestAllLightTypes:
    def test_point_light_contributes(self):
        ls = LightingSystem()
        ls.lights.append(Light("p1", "point", (0,0,0), 1.0))
        result = ls.calculate_lighting((1,0,0), (1,0,0))
        assert result > 0.1

class TestShadowQuality:
    def test_shadow_uses_pcf(self):
        ls = LightingSystem()
        result = ls.render_shadow_map("l1", 2048)
        assert result == True

class TestIndirectLight:
    def test_gi_includes_indirect(self):
        ls = LightingSystem()
        result = ls.calculate_gi((0,0,0))
        assert result > 0.1

class TestBakeUV:
    def test_bake_detects_uv_overlap(self):
        ls = LightingSystem()
        uv_data = [(0,0,1,1), (0,0,1,1)]
        result = ls.bake_lighting(uv_data)
        assert len(result) < len(uv_data)

class TestAttenuation:
    def test_attenuation_physical(self):
        ls = LightingSystem()
        d1 = ls.calculate_attenuation(1, 10)
        d2 = ls.calculate_attenuation(5, 10)
        assert d1 > d2 * 2
