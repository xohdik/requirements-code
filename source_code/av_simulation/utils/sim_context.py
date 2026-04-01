
from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

# MetaDrive
from metadrive import MultiAgentMetaDrive

# Panda3D physics
from panda3d.bullet import BulletBoxShape, BulletRigidBodyNode
from panda3d.core import (
    Camera,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    NodePath,
    PerspectiveLens,
    Point3,
    Vec3,
    WindowProperties,
)

# Local config — import the *module* so OBSTACLE_POSITION mutations are visible
from av_simulation.config import simulation_config as cfg

_front_cameras: dict = {}

class PerformanceProfiler:
    """Optional wall-time profiler for sim-step, VLM, and MPC timings.

    Pass ``enabled=False`` (default when ``--profile`` is not set) to make
    all methods no-ops with zero overhead.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled   = enabled
        self._t0       = self._vlm_t0 = self._mpc_t0 = 0.0
        self.step_times: List[float] = []
        self.vlm_times:  List[float] = []
        self.mpc_times:  List[float] = []

    def start_step(self) -> None:
        if self.enabled:
            self._t0 = time.perf_counter()

    def end_step(self) -> None:
        if self.enabled:
            self.step_times.append(time.perf_counter() - self._t0)

    def start_vlm(self) -> None:
        if self.enabled:
            self._vlm_t0 = time.perf_counter()

    def end_vlm(self) -> None:
        if self.enabled:
            self.vlm_times.append(time.perf_counter() - self._vlm_t0)

    def start_mpc(self) -> None:
        if self.enabled:
            self._mpc_t0 = time.perf_counter()

    def end_mpc(self) -> None:
        if self.enabled:
            self.mpc_times.append(time.perf_counter() - self._mpc_t0)

    def summary(self) -> None:
        """Print a formatted performance summary to stdout."""
        if not self.enabled or not self.step_times:
            return

        def _s(arr: list, lbl: str) -> None:
            a = np.array(arr)
            print(
                f"  {lbl:<22} | mean={a.mean() * 1000:6.1f}ms  "
                f"max={a.max() * 1000:6.1f}ms  n={len(a)}"
            )

        print("\n" + "=" * 60)
        print("[PROFILER] Performance Summary")
        print("=" * 60)
        print(f"  Sim FPS: {1.0 / np.mean(self.step_times):.1f}")
        _s(self.step_times, "Step wall-time")
        if self.vlm_times:
            _s(self.vlm_times, "VLM latency")
        if self.mpc_times:
            _s(self.mpc_times, "MPC solve time")
        print("=" * 60 + "\n")

class EnhancedMultiAgentEnv(MultiAgentMetaDrive):
    """MetaDrive multi-agent environment with lidar + depth fusion support."""

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({"use_semantic": False, "use_depth": False})
        return config

    @staticmethod
    def fuse_lidar_depth(
        lidar: np.ndarray, depth: Optional[np.ndarray]
    ) -> np.ndarray:
        """Element-wise minimum of lidar and column-projected depth values.

        Args:
            lidar: 1-D array of normalised lidar readings (0–1).
            depth: Optional H×W depth image; ignored when None or empty.

        Returns:
            Fused 1-D array the same length as *lidar*.
        """
        if depth is None or depth.size == 0:
            return lidar
        h, w     = depth.shape[:2]
        n        = len(lidar)
        depth_1d = depth.min(axis=0)
        indices  = np.linspace(0, w - 1, n).astype(int)
        return np.minimum(lidar, np.clip(depth_1d[indices], 0, 1))

class MultiObstacleManager:
    """Spawns one or more coloured Bullet rigid-body boxes as road obstacles.

    After ``spawn_all`` is called, ``cfg.OBSTACLE_POSITION`` is updated to the
    first obstacle's world position so that other modules can reference it via
    the module-level constant.
    """

    OBSTACLE_TYPES = [
        {"name": "crate",   "half": (1.20, 1.60, 0.90), "color": (1.0, 0.0, 0.0, 1.0)},
        {"name": "debris",  "half": (1.60, 1.50, 0.50), "color": (1.0, 0.6, 0.0, 1.0)},
        {"name": "barrier", "half": (0.60, 1.60, 0.70), "color": (1.0, 1.0, 0.0, 1.0)},
    ]

    def __init__(self, num_obstacles: int = 1, seed: int = 42) -> None:
        self.num_obstacles    = num_obstacles
        self.rng              = np.random.default_rng(seed)
        self.obstacle_bodies: list                  = []
        self.obstacle_positions: List[list]         = []

    def compute_fleet_cost(self, agents_positions: Dict[str, list]) -> float:
        """Mean inverse-distance cost from every agent to the nearest obstacle."""
        if not self.obstacle_positions or not agents_positions:
            return 0.0
        total = 0.0
        for pos in agents_positions.values():
            p       = np.array(pos[:2])
            nearest = min(
                np.linalg.norm(p - np.array(op)) for op in self.obstacle_positions
            )
            total += min(1.0, 10.0 / max(nearest, 0.1))
        return total / max(len(agents_positions), 1)

    def spawn_all(self, env, obstacle_lane) -> List[list]:
        """Spawn all obstacles along *obstacle_lane* and update cfg.OBSTACLE_POSITION."""
        longitudes = [cfg.OBSTACLE_LONGITUDE]
        for sp in self.rng.uniform(20, 35, size=max(0, self.num_obstacles - 1)):
            longitudes.append(longitudes[-1] + sp)

        for idx, lon in enumerate(longitudes):
            pos = self._spawn_single(
                env, obstacle_lane, lon,
                self.OBSTACLE_TYPES[idx % len(self.OBSTACLE_TYPES)],
            )
            if pos:
                self.obstacle_positions.append(pos)

        if self.obstacle_positions:
            # Update the module-level singleton so other modules see the change
            cfg.OBSTACLE_POSITION = self.obstacle_positions[0]
            print(
                f"[NEW-3] {len(self.obstacle_positions)} obstacle(s): "
                f"{self.obstacle_positions}"
            )
        return self.obstacle_positions

    def _spawn_single(
        self, env, lane, lon: float, otype: dict
    ) -> Optional[list]:
        raw_bw = _get_bullet_world(env)
        if raw_bw is None:
            return None
        hl, hw, hh = otype["half"]
        color      = otype["color"]
        md_center  = lane.position(lon, lateral=0)
        wx, wy     = float(md_center[0]), float(md_center[1])
        road_z     = _raycast_ground_z(env, wx, wy) or 0.0
        wz         = road_z + hh

        heading_deg = 0.0
        try:
            heading_deg = float(np.degrees(lane.heading_theta_at(lon)))
        except Exception:
            pass

        shape = BulletBoxShape(Vec3(hl, hw, hh))
        body  = BulletRigidBodyNode(f"obstacle_{otype['name']}_{int(lon)}")
        body.addShape(shape)
        body.setMass(0)
        np_node = env.engine.render.attachNewNode(body)
        np_node.setPos(wx, wy, wz)
        np_node.setH(heading_deg)
        _apply_obstacle_colour(env, np_node, hl, hw, hh, color)
        raw_bw.attachRigidBody(body)
        self.obstacle_bodies.append((body, np_node))
        print(
            f"[NEW-3] Spawned '{otype['name']}' lon={lon:.1f} "
            f"-> ({wx:.2f},{wy:.2f},{wz:.2f})"
        )
        return [wx, wy]

class WaypointPlanner:
    """Sigmoidal bypass/reform waypoint generator with ASCII mini-map.

    plan() builds a smooth lateral-merge trajectory around a single obstacle.
    render_ascii_map() produces a printable top-down bird's-eye view of the
    current fleet and obstacle layout.
    """

    def __init__(self, resolution: float = 2.0, lookahead: float = 60.0) -> None:
        self.resolution = resolution
        self.lookahead  = lookahead

    def plan(
        self,
        start_pos,
        obstacle_pos,
        lateral_target: float,
        target_speed: float,
        slot_index: int = 0,
        headway: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """Generate (x, y, speed) waypoints for a bypass manoeuvre.

        Args:
            start_pos:      [x, y] agent start position.
            obstacle_pos:   [x, y] obstacle world position.
            lateral_target: signed lateral offset (m) for the bypass lane.
            target_speed:   nominal longitudinal speed (m/s).
            slot_index:     platoon order index (0 = leader).  Each follower's
                            merge start is pushed back by ``slot_index * headway``
                            so that trajectories are spatially staggered.
            headway:        longitudinal platoon spacing (m).  Typically
                            cfg.PLATOON_SPACING.

        Returns:
            List of (x, y, speed) tuples at ``resolution``-metre intervals.
        """
        obs_x = obstacle_pos[0]
        
        merge_end  = obs_x - 35.0 - slot_index * headway
        bypass_end = obs_x + 20.0
        reform_end = bypass_end + 30.0
        total_end  = reform_end + 10.0

        waypoints: List[Tuple[float, float, float]] = []
        for x in np.arange(start_pos[0], total_end, self.resolution):
            if x < merge_end:
                y, speed = start_pos[1], target_speed

            elif x < bypass_end:
                t   = np.clip((x - merge_end) / max(bypass_end - merge_end, 1.0), 0, 1)
                sig = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
                y   = start_pos[1] + lateral_target * sig
                speed = target_speed * 0.8

            elif x < reform_end:
                t   = np.clip((x - bypass_end) / max(reform_end - bypass_end, 1.0), 0, 1)
                sig = 1.0 / (1.0 + np.exp(-10 * (t - 0.5)))
                y   = lateral_target * (1.0 - sig)
                speed = target_speed * 0.9

            else:
                y, speed = 0.0, target_speed

            waypoints.append((float(x), float(y), float(speed)))

        return waypoints

    @staticmethod
    def render_ascii_map(
        agents_positions: Dict[str, list],
        obstacle_positions: List[list],
        width: int = 80,
        height: int = 12,
    ) -> str:
        """Render a simple ASCII bird's-eye map to a string.

        Returns an empty string if there are no agent positions.
        """
        if not agents_positions:
            return ""

        all_x = (
            [p[0] for p in agents_positions.values()]
            + [p[0] for p in obstacle_positions]
        )
        all_y = (
            [p[1] for p in agents_positions.values()]
            + [p[1] for p in obstacle_positions]
        )
        x_min, x_max = min(all_x) - 5, max(all_x) + 15
        y_min, y_max = min(all_y) - 8, max(all_y) + 8
        grid = [['·'] * width for _ in range(height)]

        def w2g(wx: float, wy: float) -> Tuple[int, int]:
            gx = int((wx - x_min) / (x_max - x_min) * (width  - 1))
            gy = int((wy - y_min) / (y_max - y_min) * (height - 1))
            return int(np.clip(gx, 0, width - 1)), int(np.clip(gy, 0, height - 1))

        for op in obstacle_positions:
            gx, gy = w2g(op[0], op[1])
            grid[gy][gx] = '█'

        for i, (aid, pos) in enumerate(sorted(agents_positions.items())):
            gx, gy = w2g(pos[0], pos[1])
            grid[gy][gx] = str(i)

        lines  = ["┌" + "─" * width + "┐"]
        lines += ["│" + "".join(row) + "│" for row in reversed(grid)]
        lines += ["└" + "─" * width + "┘"]
        lines.append(f"  agents=0..{len(agents_positions) - 1}  █=obstacle")
        return "\n".join(lines)

class RealWorldDataLoader:
    """Thin stub for injecting Waymo / nuScenes configs into the env."""

    @staticmethod
    def get_env_config(args) -> dict:
        """Return extra env_config kwargs for real-world dataset loading.

        Falls back silently to an empty dict if dependencies are missing.
        """
        if args.waymo:
            try:
                from metadrive.envs.scenario_env import ScenarioEnv  # noqa: F401
                print("[NEW-7] Waymo dataset requested.")
                return {"data_directory": "waymo_data", "num_scenarios": 10}
            except ImportError:
                warnings.warn("[NEW-7] ScenarioEnv unavailable; using synthetic map.")
        if args.nuscenes:
            print("[NEW-7] nuScenes stub active; using synthetic map.")
        return {}

def attach_front_camera(env, vehicle) -> None:
    """Attach an offscreen front-facing camera to *vehicle*.

    Camera info is stored in the module-level ``_front_cameras`` registry
    keyed by ``vehicle.name``.
    """
    agent_id  = vehicle.name
    fb_props  = FrameBufferProperties()
    fb_props.setRgbColor(True)
    fb_props.setDepthBits(16)
    win_props = WindowProperties.size(cfg.FRONT_CAM_W, cfg.FRONT_CAM_H)
    buffer = env.engine.graphics_engine.makeOutput(
        env.engine.pipe,
        f"front_cam_{agent_id[:8]}",
        -100,
        fb_props,
        win_props,
        GraphicsPipe.BFRefuseWindow,
        env.engine.win.getGsg(),
        env.engine.win,
    )
    if buffer is None:
        print(f"[CAM] WARNING: no offscreen buffer for {agent_id[:8]}.")
        return

    from panda3d.core import Texture
    tex = Texture(f"front_tex_{agent_id[:8]}")
    tex.setFormat(Texture.FRgb)
    buffer.addRenderTexture(tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor)

    cam_np = env.engine.makeCamera(buffer)
    lens   = PerspectiveLens()
    lens.setFov(cfg.FRONT_CAM_FOV)
    lens.setNear(0.5)
    lens.setFar(200.0)
    cam_np.node().setLens(lens)
    cam_np.reparentTo(vehicle.origin)
    cam_np.setPos(*cfg.FRONT_CAM_OFFSET)
    cam_np.setHpr(0, -10, 0)

    env.engine.graphics_engine.renderFrame()
    env.engine.graphics_engine.renderFrame()

    _front_cameras[agent_id] = {"buffer": buffer, "texture": tex, "cam_np": cam_np}
    print(f"[CAM] Front camera attached to {agent_id[:8]}")


def capture_agent_frame(env, vehicle) -> PILImage.Image:
    """Capture a PIL RGB image from the vehicle's front camera (or main cam).

    Falls back to the main window screenshot when no offscreen buffer exists.
    The raw image is also saved to ``debug_frame_<id>.jpg`` for inspection.
    """
    agent_id = vehicle.name
    cam_info = _front_cameras.get(agent_id)

    if cam_info is not None:
        env.engine.graphics_engine.renderFrame()
        tex = cam_info["texture"]
        if not tex.hasRamImage():
            return PILImage.new("RGB", (cfg.FRONT_CAM_W, cfg.FRONT_CAM_H))
        data = tex.getRamImageAs("RGB")
        arr  = np.frombuffer(bytes(data), dtype=np.uint8).copy()
        arr  = np.flipud(arr.reshape((tex.getYSize(), tex.getXSize(), 3)))
        img  = PILImage.fromarray(arr)
    else:
        cam    = getattr(env.engine, 'main_camera', None)
        orig_v = None
        if cam is not None:
            orig_v = getattr(cam, '_current_track', None)
            cam.track(vehicle)
            env.engine.graphicsEngine.renderFrame()
        win  = env.engine.win
        tss  = win.getDisplayRegion(0).getScreenshot()
        data = tss.getRamImageAs("RGB")
        arr  = np.frombuffer(bytes(data), dtype=np.uint8).copy()
        arr  = np.flipud(arr.reshape((tss.getYSize(), tss.getXSize(), 3)))
        if cam and orig_v:
            cam.track(orig_v)
            env.engine.graphicsEngine.renderFrame()
        img = PILImage.fromarray(arr)

    try:
        img.save(f"debug_frame_{agent_id[:8]}.jpg")
    except Exception:
        pass
    return img


def detach_all_front_cameras() -> None:
    """Remove every registered front camera node and clear the registry."""
    for agent_id, cam_info in _front_cameras.items():
        try:
            cam_info["cam_np"].removeNode()
            cam_info["buffer"].clearRenderTextures()
        except Exception as e:
            print(f"[CAM] Cleanup warning {agent_id[:8]}: {e}")
    _front_cameras.clear()
    print("[CAM] All front cameras detached.")


def get_agent_spawn_lane(env):
    """Return the lane object that the first active agent is driving on.

    Tries three progressively deeper fallbacks before returning None.
    """
    for aid, veh in env.agent_manager.active_agents.items():
        if hasattr(veh, 'lane') and veh.lane is not None:
            return veh.lane
        if (
            hasattr(veh, 'navigation')
            and veh.navigation is not None
            and hasattr(veh.navigation, 'current_lane')
            and veh.navigation.current_lane is not None
        ):
            return veh.navigation.current_lane
    try:
        return env.current_map.road_network.get_lane(cfg.SPAWN_LANE_INDEX)
    except Exception as e:
        print(f"[DIAG] Lane lookup failed: {e}")
    return None


def assign_vlm_agent(policies: dict, positions: dict) -> None:
    """Designate the frontmost (highest x) agent as the VLM observer.

    All other agents fall back to lidar-only detection.
    """
    if not policies or not positions:
        return
    frontmost = max(
        [aid for aid in policies if aid in positions],
        key=lambda a: positions[a][0],
        default=None,
    )
    for aid, policy in policies.items():
        policy._is_vlm_agent = (aid == frontmost)
    if frontmost:
        print(f"[STAGE 1] VLM observer: {frontmost[:8]}")


def spawn_obstacle(env, obstacle_lane) -> List[list]:
    """Convenience wrapper: create a single-obstacle manager and spawn it.

    Returns the list of obstacle positions (length 1 for default usage).
    """
    mgr = MultiObstacleManager(num_obstacles=1)
    return mgr.spawn_all(env, obstacle_lane)


def _get_bullet_world(env):
    """Locate the BulletWorld object inside the MetaDrive physics engine.

    Tries a set of known attribute names before falling back to a full dir()
    scan, making it robust across MetaDrive versions.

    Returns None if no BulletWorld can be found (offscreen / mock envs).
    """
    pw = env.engine.physics_world
    for attr in ('dynamic_world', '_dynamic_world', 'bullet_world'):
        raw = getattr(pw, attr, None)
        if raw is not None:
            return raw
    for attr in dir(pw):
        if attr.startswith('_'):
            continue
        try:
            obj = getattr(pw, attr)
            if 'BulletWorld' in type(obj).__name__:
                return obj
        except Exception:
            continue
    return None


def _raycast_ground_z(env, wx: float, wy: float) -> Optional[float]:
    """Return the ground-surface Z at world position (wx, wy) via raycasting.

    Returns None when no Bullet world is available.
    """
    raw_bw = _get_bullet_world(env)
    if raw_bw is None:
        return None
    result = raw_bw.rayTestClosest(Point3(wx, wy, 50.0), Point3(wx, wy, -10.0))
    return result.getHitPos()[2] if result.hasHit() else None


def _build_box_geom(hl: float, hw: float, hh: float, color: tuple):
    """Build a coloured Panda3D GeomNode box (6 faces, 2 triangles each)."""
    from panda3d.core import (
        Geom, GeomNode, GeomTriangles, GeomVertexData,
        GeomVertexFormat, GeomVertexWriter,
    )
    r, g, b, a = color
    fmt   = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData("box_vdata", fmt, Geom.UHStatic)
    vdata.setNumRows(24)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")

    faces = [
        ([(hl, -hw, -hh), (hl,  hw, -hh), (hl,  hw,  hh), (hl, -hw,  hh)],  ( 1,  0,  0)),
        ([(-hl, hw, -hh), (-hl,-hw, -hh), (-hl,-hw,  hh), (-hl, hw,  hh)],  (-1,  0,  0)),
        ([(-hl,-hw, -hh), ( hl,-hw, -hh), ( hl,-hw,  hh), (-hl,-hw,  hh)],  ( 0, -1,  0)),
        ([( hl, hw, -hh), (-hl, hw, -hh), (-hl, hw,  hh), ( hl, hw,  hh)],  ( 0,  1,  0)),
        ([(-hl,-hw,  hh), ( hl,-hw,  hh), ( hl, hw,  hh), (-hl, hw,  hh)],  ( 0,  0,  1)),
        ([( hl, hw, -hh), (-hl, hw, -hh), (-hl,-hw, -hh), ( hl,-hw, -hh)],  ( 0,  0, -1)),
    ]
    tris = GeomTriangles(Geom.UHStatic)
    vi   = 0
    for corners, norm in faces:
        for cx, cy, cz in corners:
            vw.addData3f(cx, cy, cz)
            nw.addData3f(*norm)
            cw.addData4f(r, g, b, a)
        tris.addVertices(vi, vi + 1, vi + 2)
        tris.addVertices(vi, vi + 2, vi + 3)
        vi += 4

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("colored_box")
    node.addGeom(geom)
    return node


def _apply_obstacle_colour(
    env, np_node, hl: float, hw: float, hh: float, color: tuple
) -> None:
    """Attach a coloured visual to a Bullet rigid-body node path.

    Primary path: solid coloured GeomNode via _build_box_geom.
    Fallback:     wireframe using LineSegs when GeomNode creation fails.
    """
    from panda3d.core import ColorAttrib, LColor
    try:
        visual = np_node.attachNewNode(_build_box_geom(hl, hw, hh, color))
        visual.setAttrib(ColorAttrib.makeFlat(LColor(*color)), priority=100)
        visual.setLightOff(1)
        visual.setTextureOff(1)
    except Exception:
        try:
            from panda3d.core import LineSegs
            ls = LineSegs("obs_wire")
            ls.setThickness(6.0)
            ls.setColor(*color)
            edges = [
                ((-hl, -hw, -hh), ( hl, -hw, -hh)), (( hl, -hw, -hh), ( hl,  hw, -hh)),
                (( hl,  hw, -hh), (-hl,  hw, -hh)), ((-hl,  hw, -hh), (-hl, -hw, -hh)),
                ((-hl, -hw,  hh), ( hl, -hw,  hh)), (( hl, -hw,  hh), ( hl,  hw,  hh)),
                (( hl,  hw,  hh), (-hl,  hw,  hh)), ((-hl,  hw,  hh), (-hl, -hw,  hh)),
                ((-hl, -hw, -hh), (-hl, -hw,  hh)), (( hl, -hw, -hh), ( hl, -hw,  hh)),
                (( hl,  hw, -hh), ( hl,  hw,  hh)), ((-hl,  hw, -hh), (-hl,  hw,  hh)),
            ]
            for (ax, ay, az), (bx, by, bz) in edges:
                ls.moveTo(ax, ay, az)
                ls.drawTo(bx, by, bz)
            np_node.attachNewNode(ls.create())
        except Exception as e2:
            print(f"[PLACE] All visual attempts failed: {e2}")
