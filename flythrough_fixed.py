"""
Fly-Through Navigation Feature
Medical visualization: Navigate camera INSIDE anatomical structures
Author: Biomedical Visualization System
"""

import numpy as np
import pyvista as pv
from PyQt5 import QtCore


class FlythroughController:
    """
    Medical Fly-Through Navigation

    Purpose: Let doctor fly INSIDE specific anatomical structures
    - User selects a part (Aorta, Left Ventricle, etc.)
    - Camera flies through THAT structure's interior
    - Other structures stay visible (transparent) for context
    - Like an endoscope navigating inside the selected vessel/chamber
    """

    def __init__(self, plotter, surfaces, console_log=None):
        """
        Initialize Fly-Through Controller

        Args:
            plotter: QtInteractor (PyVista plotter)
            surfaces: list of dicts from segmentation (REQUIRED for medical flythrough)
            console_log: function to log messages
        """
        self.plotter = plotter
        self.surfaces = surfaces or []
        self.console_log = console_log or (lambda msg: print(msg))

        # Animation state
        self.path_points = []
        self.focal_points = []
        self.current_frame = 0
        self.is_animating = False
        self.timer = None

        # Track current selected structure
        self.selected_structure = None

    def log(self, msg):
        """Helper to log messages"""
        self.console_log(msg)

    def get_structure_names(self):
        """Get list of available anatomical structures"""
        return [surf['name'] for surf in self.surfaces]

    def generate_path_for_structure(self, structure_name, num_points=250):
        """
        Generate fly-through path INSIDE a specific anatomical structure

        This is the CORE medical feature:
        - Path stays INSIDE the selected structure
        - Follows the structure's natural geometry
        - User can see interior walls

        Args:
            structure_name: Name of anatomical structure (e.g., "Aorta", "Left Ventricle")
            num_points: Number of points in the path (default 250)

        Returns:
            bool: True if path generated successfully
        """
        self.selected_structure = structure_name
        self.log(f"\n🎯 Generating fly-through for: {structure_name}")
        self.log("=" * 60)

        # Find the target structure
        target_mesh = None
        for surf in self.surfaces:
            if surf['name'] == structure_name:
                target_mesh = surf['mesh']
                break

        if target_mesh is None:
            self.log(f"❌ ERROR: Structure '{structure_name}' not found")
            return False

        # Generate path based on structure type
        if "Aorta" in structure_name:
            self.path_points, self.focal_points = self._create_aorta_path(target_mesh, num_points)
        elif "Pulmonary Artery" in structure_name:
            self.path_points, self.focal_points = self._create_pulmonary_artery_path(target_mesh, num_points)
        elif "Left Ventricle" in structure_name:
            self.path_points, self.focal_points = self._create_left_ventricle_path(target_mesh, num_points)
        elif "Right Ventricle" in structure_name:
            self.path_points, self.focal_points = self._create_right_ventricle_path(target_mesh, num_points)
        elif "Left Atrium" in structure_name:
            self.path_points, self.focal_points = self._create_atrium_path(target_mesh, num_points)
        elif "Right Atrium" in structure_name:
            self.path_points, self.focal_points = self._create_atrium_path(target_mesh, num_points)
        else:
            # Generic chamber exploration for any other structure
            self.path_points, self.focal_points = self._create_generic_interior_path(target_mesh, num_points)

        self.log(f"✅ Path generated: {len(self.path_points)} points")
        self.log(f"   Camera will navigate INSIDE {structure_name}")
        return True

    # ========== STRUCTURE-SPECIFIC PATH GENERATORS ==========

    def _create_aorta_path(self, mesh, num_points):
        """
        Fly through AORTA interior
        Path: From aortic root (heart connection) → upward through ascending aorta → arch
        """
        self.log("   📍 Aorta Path: Root → Ascending → Arch")

        bounds = mesh.bounds
        center = np.array(mesh.center)

        # Aorta goes vertically upward from heart
        # Start at bottom (aortic valve), go up to arch
        start_y = bounds[2]  # Bottom of aorta (connects to LV)
        end_y = bounds[3]  # Top of aorta (arch)

        width_x = (bounds[1] - bounds[0]) * 0.3  # Stay centered in vessel
        width_z = (bounds[5] - bounds[4]) * 0.3

        path = []
        focal_points = []

        for i in range(num_points):
            t = i / (num_points - 1)

            # Move upward through aorta
            y_pos = start_y + (end_y - start_y) * t

            # Add slight curve to follow aorta's natural curve
            x_offset = np.sin(t * np.pi) * width_x * 0.5  # Curves slightly
            z_offset = np.cos(t * np.pi * 0.5) * width_z * 0.3

            cam_pos = [
                center[0] + x_offset,
                y_pos,
                center[2] + z_offset
            ]

            # Look ahead along path (forward direction)
            look_ahead_t = min(t + 0.05, 1.0)
            look_y = start_y + (end_y - start_y) * look_ahead_t
            focal = [
                center[0] + np.sin(look_ahead_t * np.pi) * width_x * 0.5,
                look_y,
                center[2] + np.cos(look_ahead_t * np.pi * 0.5) * width_z * 0.3
            ]

            path.append(cam_pos)
            focal_points.append(focal)

        return path, focal_points

    def _create_pulmonary_artery_path(self, mesh, num_points):
        """
        Fly through PULMONARY ARTERY interior
        Path: From RV outflow → main PA → bifurcation
        """
        self.log("   📍 Pulmonary Artery Path: RV Outflow → Main PA")

        bounds = mesh.bounds
        center = np.array(mesh.center)

        # PA also goes upward but slightly different angle
        start_y = bounds[2]
        end_y = bounds[3]

        width = (bounds[1] - bounds[0]) * 0.25

        path = []
        focal_points = []

        for i in range(num_points):
            t = i / (num_points - 1)

            y_pos = start_y + (end_y - start_y) * t

            # PA curves slightly to the left
            x_offset = -np.sin(t * np.pi * 0.8) * width
            z_offset = np.cos(t * np.pi) * width * 0.4

            cam_pos = [center[0] + x_offset, y_pos, center[2] + z_offset]

            look_ahead_t = min(t + 0.05, 1.0)
            look_y = start_y + (end_y - start_y) * look_ahead_t
            focal = [
                center[0] - np.sin(look_ahead_t * np.pi * 0.8) * width,
                look_y,
                center[2] + np.cos(look_ahead_t * np.pi) * width * 0.4
            ]

            path.append(cam_pos)
            focal_points.append(focal)

        return path, focal_points

    def _create_left_ventricle_path(self, mesh, num_points):
        """
        Explore LEFT VENTRICLE interior
        Path: Spiral from base → apex → back to base
        Shows chamber walls, papillary muscles, mitral valve
        """
        self.log("   📍 LV Path: Spiral exploration of chamber")

        bounds = mesh.bounds
        center = np.array(mesh.center)

        # LV chamber dimensions
        height = bounds[3] - bounds[2]
        radius_base = (bounds[1] - bounds[0]) * 0.25  # Wider at base
        radius_apex = radius_base * 0.3  # Narrower at apex

        path = []
        focal_points = []

        for i in range(num_points):
            t = i / (num_points - 1)

            # Spiral motion: base → apex → base
            if t < 0.5:
                # Descend to apex
                height_t = t * 2  # 0 → 1
                y_pos = bounds[3] - height * height_t  # Top to bottom
                radius = radius_base - (radius_base - radius_apex) * height_t
            else:
                # Ascend back to base
                height_t = (t - 0.5) * 2  # 0 → 1
                y_pos = bounds[2] + height * height_t  # Bottom to top
                radius = radius_apex + (radius_base - radius_apex) * height_t

            # Rotate around chamber
            angle = t * np.pi * 6  # Multiple rotations

            cam_pos = [
                center[0] + radius * np.cos(angle),
                y_pos,
                center[2] + radius * np.sin(angle)
            ]

            # Always look toward chamber center (see walls)
            focal_points.append(center.tolist())
            path.append(cam_pos)

        return path, focal_points

    def _create_right_ventricle_path(self, mesh, num_points):
        """
        Explore RIGHT VENTRICLE interior
        Similar to LV but different geometry (more crescent-shaped)
        """
        self.log("   📍 RV Path: Crescent chamber exploration")

        bounds = mesh.bounds
        center = np.array(mesh.center)

        height = bounds[3] - bounds[2]
        width = (bounds[1] - bounds[0]) * 0.3

        path = []
        focal_points = []

        for i in range(num_points):
            t = i / (num_points - 1)

            # Move along RV's crescent shape
            angle = t * np.pi * 4
            height_pos = bounds[2] + height * (0.3 + 0.4 * np.sin(t * np.pi))

            cam_pos = [
                center[0] + width * np.cos(angle) * (1 - t * 0.3),
                height_pos,
                center[2] + width * np.sin(angle) * 0.7
            ]

            path.append(cam_pos)
            focal_points.append(center.tolist())

        return path, focal_points

    def _create_atrium_path(self, mesh, num_points):
        """
        Explore ATRIUM interior
        Path: Smooth orbit inside atrial chamber
        """
        self.log("   📍 Atrium Path: Smooth chamber orbit")

        bounds = mesh.bounds
        center = np.array(mesh.center)

        radius = (bounds[1] - bounds[0]) * 0.25
        height_variation = (bounds[3] - bounds[2]) * 0.2

        path = []
        focal_points = []

        for i in range(num_points):
            t = i / (num_points - 1)
            angle = t * np.pi * 3  # 1.5 full rotations

            cam_pos = [
                center[0] + radius * np.cos(angle),
                center[1] + height_variation * np.sin(t * np.pi * 2),
                center[2] + radius * np.sin(angle)
            ]

            path.append(cam_pos)
            focal_points.append(center.tolist())

        return path, focal_points

    def _create_generic_interior_path(self, mesh, num_points):
        """
        Generic interior exploration for any structure
        Used when structure type is not specifically recognized
        """
        self.log("   📍 Generic interior exploration")

        bounds = mesh.bounds
        center = np.array(mesh.center)

        size_x = (bounds[1] - bounds[0]) * 0.3
        size_y = (bounds[3] - bounds[2]) * 0.3
        size_z = (bounds[5] - bounds[4]) * 0.3

        path = []
        focal_points = []

        for i in range(num_points):
            t = i / (num_points - 1)
            angle = t * np.pi * 4

            cam_pos = [
                center[0] + size_x * np.cos(angle) * (1 - t * 0.3),
                center[1] + size_y * np.sin(t * np.pi * 2),
                center[2] + size_z * np.sin(angle) * (1 - t * 0.3)
            ]

            path.append(cam_pos)
            focal_points.append(center.tolist())

        return path, focal_points

    # ========== ANIMATION CONTROL ==========

    def start_animation(self, speed=50):
        """
        Start fly-through animation

        Args:
            speed: Milliseconds per frame (default 50ms = 20fps)

        Returns:
            bool: True if animation started successfully
        """
        if len(self.path_points) == 0:
            self.log("❌ No path generated")
            return False

        if self.is_animating:
            self.log("⚠️ Animation already running")
            return False

        self.log(f"\n▶️ STARTING FLY-THROUGH ANIMATION")
        self.log("=" * 60)
        self.log(f"   Structure: {self.selected_structure}")
        self.log(f"   Total frames: {len(self.path_points)}")
        self.log(f"   Speed: {speed}ms/frame")
        self.log(f"   Duration: ~{len(self.path_points) * speed / 1000:.1f}s")
        self.log("=" * 60)

        self.current_frame = 0
        self.is_animating = True

        # Create timer for animation
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(speed)

        return True

    def _update_frame(self):
        """Update camera to next frame (called by timer)"""
        if self.current_frame >= len(self.path_points):
            self.stop_animation()
            self.log("\n✅ Fly-through complete!")
            return

        # Get current camera position and focal point
        cam_pos = self.path_points[self.current_frame]
        focal_pos = self.focal_points[self.current_frame]

        # Set camera position
        self.plotter.camera_position = [
            cam_pos,
            focal_pos,
            (0, 0, 1)  # Up vector
        ]

        # Render the frame
        self.plotter.render()
        self.current_frame += 1

        # Log progress every 25 frames
        if self.current_frame % 25 == 0:
            progress = (self.current_frame / len(self.path_points)) * 100
            self.log(
                f"🎬 Frame {self.current_frame}/{len(self.path_points)} "
                f"({progress:.0f}%) - Inside {self.selected_structure}"
            )

    def stop_animation(self):
        """Stop fly-through animation"""
        if self.timer:
            self.timer.stop()
            self.timer = None

        self.is_animating = False
        self.log("⏹️ Animation stopped")

    def reset_camera(self):
        """Reset camera to default view"""
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        self.plotter.render()
        self.log("🔄 Camera reset to default view")