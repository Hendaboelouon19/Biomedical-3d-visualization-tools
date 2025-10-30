import numpy as np
import pyvista as pv
import copy
import math
from PyQt5 import QtCore


###########################################################
# Heart Pump Feature (Moving Stuff) - OPTIMIZED VERSION
# Reduced contraction + Better performance
###########################################################

class HeartPumpController:
    """
    Controller for heart pump animation with realistic cardiac cycle
    and continuous blood flow visualization - OPTIMIZED
    """

    def __init__(self, plotter, surfaces, console_log=None):
        """
        Initialize heart pump animation
        Args:
            plotter: existing PyVista or BackgroundPlotter instance.
            surfaces: list of dicts from build_heart_surfaces_from_seg
                     [{'name': 'Left Ventricle', 'color': [...], 'mesh': pv.PolyData}, ...]
            console_log: function to log messages
        """
        self.plotter = plotter
        self.surfaces = surfaces
        self.console_log = console_log or (lambda msg: print(msg))

        self.is_animating = False
        self.animation_timer = None
        self.base_points = {}
        self.base_meshes = {}
        self.t_val = {"t": 0.0}

        # Blood flow - continuous solid tubes
        self.blood_flow_actors = []
        self.blood_paths = []

        # Build name->actor mapping
        self.name_to_actor = {}
        self._build_actor_mapping()

        # Performance optimization
        self.render_counter = 0
        self.render_every_n_frames = 1  # Render every frame for smooth animation

        self.log("[Moving Stuff] Initializing heart pump animation...")

    def _build_actor_mapping(self):
        """Build mapping of structure names to actors"""
        self.name_to_actor = {}
        for surf in self.surfaces:
            name = surf['name']
            actor = self.plotter.actors.get(name)
            if actor:
                self.name_to_actor[name] = actor

    def log(self, msg):
        """Helper to log messages"""
        self.console_log(msg)

    def find_structures(self, keywords):
        """Helper for fuzzy chamber detection"""
        out = []
        for surf in self.surfaces:
            name = surf['name']
            low = name.lower()
            for word in keywords:
                if word in low:
                    out.append(name)
                    break
        return out

    def start_animation(self):
        """Start the heart pumping animation with blood flow"""
        if self.is_animating:
            self.log("⚠ Animation already running!")
            return False

        self.log("\n" + "=" * 60)
        self.log("💓 STARTING HEART PUMP ANIMATION")
        self.log("=" * 60)

        # Detect structures
        left_v = self.find_structures(["left ventricle", "lv", "ventricle left"])
        right_v = self.find_structures(["right ventricle", "rv", "ventricle right"])
        left_a = self.find_structures(["left atrium", "la", "atrium left"])
        right_a = self.find_structures(["right atrium", "ra", "atrium right"])
        aorta = self.find_structures(["aorta", "ascending", "arch", "aortic"])

        # Blood vessels
        pulmonary_artery = self.find_structures(["pulmonary artery", "pulmonary trunk", "pa", "trunk"])
        pulmonary_vein = self.find_structures(["pulmonary vein", "pv"])
        vena_cava = self.find_structures(["vena cava", "superior vena", "inferior vena", "svc", "ivc", "cava"])

        self.log(f"📋 Detected structures:")
        self.log(f"   • Left Ventricle: {left_v if left_v else 'None'}")
        self.log(f"   • Right Ventricle: {right_v if right_v else 'None'}")
        self.log(f"   • Left Atrium: {left_a if left_a else 'None'}")
        self.log(f"   • Right Atrium: {right_a if right_a else 'None'}")
        self.log(f"   • Aorta: {aorta if aorta else 'None'}")

        # Save original coordinates - ONLY for chambers that should move
        self.base_points = {}
        self.base_meshes = {}

        # Store chamber references
        self.left_v = left_v
        self.right_v = right_v
        self.left_a = left_a
        self.right_a = right_a
        self.aorta_names = aorta
        self.pulmonary_artery_names = pulmonary_artery
        self.pulmonary_vein_names = pulmonary_vein
        self.vena_cava_names = vena_cava

        # Only save base points for chambers (not vessels)
        chamber_names = left_v + right_v + left_a + right_a

        for name in chamber_names:
            surf = None
            for s in self.surfaces:
                if s['name'] == name:
                    surf = s
                    break

            if surf is None:
                continue

            try:
                mesh = surf['mesh']
                self.base_points[name] = copy.deepcopy(mesh.points)
                self.base_meshes[name] = mesh
                self.log(f"   ✓ Chamber: {name}")
            except Exception as e:
                self.log(f"   ⚠ Couldn't save {name}: {e}")

        if not self.base_points:
            self.log("❌ ERROR: No chambers found to animate!")
            return False

        # Determine coordinate system
        self.axis_index = 1  # Y-axis
        ref_name = chamber_names[0]
        ref_pts = self.base_points[ref_name]
        y_min = ref_pts[:, self.axis_index].min()
        y_max = ref_pts[:, self.axis_index].max()
        self.y_mid = 0.5 * (y_min + y_max)

        # Animation parameters
        self.t_val = {"t": 0.0}
        self.beat_period = 1.0  # 60 BPM

        self.log(f"\n⏱ Animation settings:")
        self.log(f"   Beat period: {self.beat_period} seconds (60 BPM)")
        self.log(f"   Frame rate: ~30 fps")

        # Initialize blood flow paths
        self._initialize_blood_flow()

        # Create timer with optimized interval
        self.animation_timer = QtCore.QTimer()
        self.animation_timer.timeout.connect(self._update_beat)
        self.animation_timer.start(33)  # 33ms = ~30 FPS

        self.is_animating = True

        self.log("\n" + "=" * 60)
        self.log("✅ HEART PUMP ANIMATION STARTED!")
        self.log("=" * 60)

        return True

    def _initialize_blood_flow(self):
        """Initialize blood flow paths - 6 anatomical pathways"""
        self.blood_paths = []

        # Get chamber centers
        centers = {}
        for chamber_list in [self.left_v, self.right_v, self.left_a, self.right_a]:
            if chamber_list:
                name = chamber_list[0]
                centers[name] = self._get_structure_center(name)

        # PATH 1: Vena Cava → Right Atrium
        if self.vena_cava_names and self.right_a:
            vc_pos = self._get_structure_center(self.vena_cava_names[0])
            ra_pos = centers[self.right_a[0]]
            self.blood_paths.append({
                'start': vc_pos,
                'end': ra_pos,
                'color': [0.0, 0.0, 1.0],
                'cycle_start': 0.0,
                'cycle_end': 0.25,
                'path_type': 'straight',
                'name': 'Vena Cava → RA'
            })

        # PATH 2: Right Atrium → Right Ventricle
        if self.right_a and self.right_v:
            ra_pos = centers[self.right_a[0]]
            rv_pos = centers[self.right_v[0]]
            self.blood_paths.append({
                'start': ra_pos,
                'end': rv_pos,
                'color': [0.0, 0.0, 1.0],
                'cycle_start': 0.10,
                'cycle_end': 0.35,
                'path_type': 'straight',
                'name': 'RA → RV'
            })

        # PATH 3: Right Ventricle → Pulmonary Artery
        if self.right_v and self.pulmonary_artery_names:
            rv_pos = centers[self.right_v[0]]
            pa_pos = self._get_structure_center(self.pulmonary_artery_names[0])
            self.blood_paths.append({
                'start': rv_pos,
                'end': pa_pos,
                'color': [0.0, 0.0, 1.0],
                'cycle_start': 0.35,
                'cycle_end': 0.60,
                'path_type': 'curved_up',
                'name': 'RV → Pulmonary Artery'
            })

        # PATH 4: Pulmonary Veins → Left Atrium
        if self.pulmonary_vein_names and self.left_a:
            pv_pos = self._get_structure_center(self.pulmonary_vein_names[0])
            la_pos = centers[self.left_a[0]]
            self.blood_paths.append({
                'start': pv_pos,
                'end': la_pos,
                'color': [1.0, 0.0, 0.0],
                'cycle_start': 0.0,
                'cycle_end': 0.25,
                'path_type': 'straight',
                'name': 'Pulmonary Veins → LA'
            })

        # PATH 5: Left Atrium → Left Ventricle
        if self.left_a and self.left_v:
            la_pos = centers[self.left_a[0]]
            lv_pos = centers[self.left_v[0]]
            self.blood_paths.append({
                'start': la_pos,
                'end': lv_pos,
                'color': [1.0, 0.0, 0.0],
                'cycle_start': 0.10,
                'cycle_end': 0.35,
                'path_type': 'straight',
                'name': 'LA → LV'
            })

        # PATH 6: Left Ventricle → Aorta
        if self.left_v and self.aorta_names:
            lv_pos = centers[self.left_v[0]]
            ao_pos = self._get_structure_center(self.aorta_names[0])
            self.blood_paths.append({
                'start': lv_pos,
                'end': ao_pos,
                'color': [1.0, 0.0, 0.0],
                'cycle_start': 0.35,
                'cycle_end': 0.60,
                'path_type': 'curved_up',
                'name': 'LV → Aorta'
            })

        self.log(f"🩸 Initialized {len(self.blood_paths)} blood flow pathways")

    def _get_structure_center(self, structure_name):
        """Get center point of a structure"""
        if structure_name in self.base_points:
            return np.mean(self.base_points[structure_name], axis=0)

        for surf in self.surfaces:
            if surf['name'] == structure_name:
                return np.mean(surf['mesh'].points, axis=0)

        return np.array([0.0, 0.0, 0.0])

    def _update_beat(self):
        """Callback function for animation - OPTIMIZED"""
        if not self.is_animating:
            return

        # Advance time
        self.t_val["t"] += 0.033
        t_norm = (self.t_val["t"] % self.beat_period) / self.beat_period

        # Update chambers
        self._update_chambers(t_norm)

        # Update blood flow
        self._update_blood_flow(t_norm)

        # Optimized rendering
        self.render_counter += 1
        if self.render_counter >= self.render_every_n_frames:
            self.render_counter = 0
            try:
                self.plotter.render()
            except:
                pass

    def _update_chambers(self, t_norm):
        """Update chamber contractions - REDUCED STRENGTH"""

        # Get contraction strengths
        la_strength = self.left_atrium_strength(t_norm)
        ra_strength = self.right_atrium_strength(t_norm)
        lv_strength = self.left_ventricle_strength(t_norm)
        rv_strength = self.right_ventricle_strength(t_norm)

        # Update chambers
        for name in self.left_a:
            if name in self.base_meshes and name in self.base_points:
                self._apply_contraction(name, la_strength, is_atrium=True)

        for name in self.right_a:
            if name in self.base_meshes and name in self.base_points:
                self._apply_contraction(name, ra_strength, is_atrium=True)

        for name in self.left_v:
            if name in self.base_meshes and name in self.base_points:
                self._apply_contraction(name, lv_strength, is_atrium=False)

        for name in self.right_v:
            if name in self.base_meshes and name in self.base_points:
                self._apply_contraction(name, rv_strength, is_atrium=False)

    def _apply_contraction(self, name, strength, is_atrium):
        """Apply contraction to a chamber"""
        mesh = self.base_meshes[name]
        pts0 = self.base_points[name]

        if is_atrium:
            upper_mask = pts0[:, self.axis_index] > self.y_mid
            if np.any(upper_mask):
                center = np.mean(pts0[upper_mask], axis=0)
            else:
                center = np.mean(pts0, axis=0)
        else:
            lower_mask = pts0[:, self.axis_index] <= self.y_mid
            if np.any(lower_mask):
                center = np.mean(pts0[lower_mask], axis=0)
            else:
                center = np.mean(pts0, axis=0)

        # Apply contraction
        pts = center + (pts0 - center) * (1.0 - strength)
        mesh.points = pts

    def _update_blood_flow(self, t_norm):
        """Update blood flow - continuous solid tubes"""

        # Remove old blood flow actors
        for actor in self.blood_flow_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self.blood_flow_actors = []

        # Draw each blood flow path
        for path in self.blood_paths:
            cycle_start = path['cycle_start']
            cycle_end = path['cycle_end']

            active = False
            intensity = 0.0

            if cycle_start <= t_norm < cycle_end:
                active = True
                cycle_duration = cycle_end - cycle_start
                flow_progress = (t_norm - cycle_start) / cycle_duration
                intensity = np.sin(np.pi * flow_progress)

            elif cycle_start > cycle_end:
                if t_norm >= cycle_start or t_norm < cycle_end:
                    active = True
                    if t_norm >= cycle_start:
                        flow_progress = (t_norm - cycle_start) / (1.0 - cycle_start + cycle_end)
                    else:
                        flow_progress = (1.0 - cycle_start + t_norm) / (1.0 - cycle_start + cycle_end)
                    intensity = np.sin(np.pi * flow_progress)

            if active and intensity > 0.1:
                self._draw_continuous_blood_stream(path, intensity)

    def _draw_continuous_blood_stream(self, path, intensity):
        """Draw a continuous solid blood stream tube"""
        start = path['start']
        end = path['end']
        color = path['color']
        path_type = path['path_type']

        n_segments = 30  # Reduced from 40 for better performance

        points = []
        for i in range(n_segments):
            t = i / (n_segments - 1)

            if path_type == 'curved_up':
                pos = start + t * (end - start)
                arc_height = 15.0 * np.sin(np.pi * t)
                pos[1] += arc_height
            else:
                pos = start + t * (end - start)

            points.append(pos)

        points = np.array(points)

        spline = pv.Spline(points, n_segments)

        tube_radius = 2.0 + 1.0 * intensity
        tube = spline.tube(radius=tube_radius, n_sides=12)  # Reduced from 16

        tube_color = [c * (0.8 + 0.2 * intensity) for c in color]

        actor = self.plotter.add_mesh(
            tube,
            color=tube_color,
            opacity=0.85,
            smooth_shading=True,
            specular=0.6,  # Reduced
            specular_power=25,  # Reduced
            lighting=True,
            show_edges=False
        )

        self.blood_flow_actors.append(actor)

    def stop_animation(self):
        """Stop the heart pumping animation"""
        if not self.is_animating:
            self.log("⚠ Animation is not running!")
            return False

        self.log("\n" + "=" * 60)
        self.log("⏸ STOPPING HEART PUMP ANIMATION")
        self.log("=" * 60)

        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer.deleteLater()
            self.animation_timer = None

        for actor in self.blood_flow_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self.blood_flow_actors = []

        for name, pts0 in self.base_points.items():
            if name not in self.base_meshes:
                continue
            try:
                mesh = self.base_meshes[name]
                mesh.points = pts0.copy()
                self.log(f"   ✓ Reset: {name}")
            except Exception as e:
                self.log(f"   ⚠ Error resetting {name}: {e}")

        try:
            self.plotter.render()
        except:
            pass

        self.is_animating = False

        self.log("=" * 60)
        self.log("✅ Animation stopped and heart reset")
        self.log("=" * 60 + "\n")

        return True

    # ============================================================
    # CARDIAC CYCLE TIMING - REDUCED CONTRACTION STRENGTH
    # ============================================================

    def left_atrium_strength(self, tt):
        """Left atrium - REDUCED from 20% to 12%"""
        if 0.10 <= tt < 0.25:
            phase = (tt - 0.10) / 0.15
            return 0.12 * np.sin(np.pi * phase) ** 2
        return 0.0

    def right_atrium_strength(self, tt):
        """Right atrium - REDUCED from 20% to 12%"""
        if 0.10 <= tt < 0.25:
            phase = (tt - 0.10) / 0.15
            return 0.12 * np.sin(np.pi * phase) ** 2
        return 0.0

    def left_ventricle_strength(self, tt):
        """Left ventricle - REDUCED from 35% to 18%"""
        if 0.35 <= tt < 0.60:
            phase = (tt - 0.35) / 0.25
            return 0.18 * np.sin(np.pi * phase) ** 2
        return 0.0

    def right_ventricle_strength(self, tt):
        """Right ventricle - REDUCED from 30% to 15%"""
        if 0.35 <= tt < 0.60:
            phase = (tt - 0.35) / 0.25
            return 0.15 * np.sin(np.pi * phase) ** 2
        return 0.0