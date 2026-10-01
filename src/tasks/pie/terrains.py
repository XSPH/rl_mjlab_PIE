"""Original box geometry parkour tracks composed using mjlab's terrain API."""

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


@dataclass(kw_only=True)
class ParkourTerrainCfg(SubTerrainCfg):
    kind: str = "flat"
    max_gap: float = 1.0
    max_obstacle_height: float = 0.75
    max_stair_height: float = 0.25

    def function(self, difficulty: float, spec: mujoco.MjSpec,
                 rng: np.random.Generator) -> TerrainOutput:
        length, width = self.size
        body = spec.body("terrain")
        geometries: list[TerrainGeometry] = []

        def box(x0: float, x1: float, y0: float, y1: float,
                z0: float, z1: float) -> None:
            if x1 <= x0 or y1 <= y0 or z1 <= z0:
                return
            geom = body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                size=((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2),
                group=0, friction=(1.0, 0.005, 0.0001),
            )
            geometries.append(TerrainGeometry(geom=geom, color=(0.30, 0.42, 0.48, 1.0)))

        # Initial and end platforms stay flat at all difficulty levels.
        start = 1.8
        finish = length - 1.0
        if self.kind == "gaps":
            # Real missing support down to -1m: do not add a hidden plane at z=0.
            pit_depth = float(rng.uniform(0.7, 1.3))
            box(0, length, 0, width, -pit_depth - 0.2, -pit_depth)
            nominal_gap = 0.08 + difficulty * (self.max_gap - 0.08)
            cursor = 0.0
            for gap_center in np.arange(start + 0.6, finish - 0.1, 2.0):
                gap = float(np.clip(nominal_gap * rng.uniform(0.85, 1.15), 0.05, self.max_gap))
                gap_start, gap_end = gap_center - gap / 2, gap_center + gap / 2
                box(cursor, gap_start, 0, width, -pit_depth, 0)
                cursor = gap_end
            box(cursor, length, 0, width, -pit_depth, 0)
        elif self.kind == "stairs":
            box(0, length, 0, width, -0.1, 0)
            height = 0.02 + difficulty * (self.max_stair_height - 0.02)
            step_width = 0.40
            centers = np.arange(start, finish, step_width)
            for index, x0 in enumerate(centers):
                # Up then down; three ascending stairs require clearance throughout.
                level = min(index + 1, max(1, len(centers) - index), 3)
                box(x0, min(x0 + step_width, finish), 0, width, 0, level * height)
        elif self.kind in {"steps", "hurdles"}:
            box(0, length, 0, width, -0.1, 0)
            height = 0.025 + difficulty * (self.max_obstacle_height - 0.025)
            run = 0.65 if self.kind == "steps" else 0.15
            for x0 in np.arange(start + 0.4, finish - run, 1.8):
                box(x0, x0 + run, 0, width, 0, height)
        elif self.kind == "flat":
            box(0, length, 0, width, -0.1, 0)
        else:
            raise ValueError(f"Unsupported parkour terrain {self.kind}")
        return TerrainOutput(origin=np.array((0.8, width / 2, 0.0)), geometries=geometries)
