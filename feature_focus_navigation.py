import numpy as np
import pyvista as pv


class FocusNavigationController:
    """
    Focus Navigation - Select structure from dropdown to focus

    Features:
    - Select heart structure from dropdown menu
    - Automatically zoom camera to selected structure
    - Make other structures transparent (0.15 opacity)
    - Keep selected structure fully opaque (0.95 opacity)
    """

    def __init__(self, plotter, surfaces, console_log=None):
        """
        plotter: QtInteractor (PyVista plotter)
        surfaces: list of dicts from build_heart_surfaces_from_seg
                  [{'name': 'Left Ventricle', 'color': [...], 'mesh': pv.PolyData}, ...]
        console_log: function to log messages
        """
        self.plotter = plotter
        self.surfaces = surfaces
        self.console_log = console_log or (lambda msg: print(msg))

        # Track current focus state
        self.current_focus = None

        # Opacity values
        self.FOCUSED_OPACITY = 0.95
        self.UNFOCUSED_OPACITY = 0.15
        self.NORMAL_OPACITY = 0.9

        # Store name->actor mapping
        self.name_to_actor = {}
        self._build_actor_mapping()

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

    def focus_on_structure(self, structure_name):
        """
        Focus camera on a specific heart structure

        Steps:
        1. Find the structure in surfaces list
        2. Set selected structure opacity to FOCUSED_OPACITY
        3. Set all other structures opacity to UNFOCUSED_OPACITY
        4. Calculate bounding box of selected structure
        5. Zoom camera to that bounding box
        """

        # Find the structure
        target_surface = None
        for surf in self.surfaces:
            if surf['name'] == structure_name:
                target_surface = surf
                break

        if target_surface is None:
            self.log(f"❌ ERROR: Structure '{structure_name}' not found!")
            return False

        self.log(f"🔍 Focusing on: {structure_name}")
        self.log("=" * 50)
        self.current_focus = structure_name

        # Update all mesh opacities
        for surf in self.surfaces:
            name = surf['name']

            # Find actor by name
            actor = self.name_to_actor.get(name)
            if actor is None:
                self.log(f"⚠️ WARNING: Actor for '{name}' not found")
                continue

            if name == structure_name:
                # Selected structure: fully visible
                actor.GetProperty().SetOpacity(self.FOCUSED_OPACITY)
                self.log(f"  ✓ {name}: opacity = {self.FOCUSED_OPACITY} ⭐ FOCUSED")
            else:
                # Other structures: transparent/faded
                actor.GetProperty().SetOpacity(self.UNFOCUSED_OPACITY)
                self.log(f"  ○ {name}: opacity = {self.UNFOCUSED_OPACITY}")

        # Zoom camera to structure's bounding box
        target_mesh = target_surface['mesh']
        bounds = target_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)

        # Calculate center and size
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]

        # Calculate diagonal size for camera distance
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]
        size = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # Set camera position (slightly offset from center)
        camera_distance = size * 1.8  # 1.8x the size for good view

        # Calculate camera position (offset from center)
        camera_pos = [
            center[0] + camera_distance * 0.5,
            center[1] + camera_distance * 0.5,
            center[2] + camera_distance * 0.8
        ]

        # Update camera smoothly
        self.plotter.camera_position = [
            camera_pos,  # camera position
            center,  # focal point (look at structure center)
            (0, 0, 1)  # view up vector
        ]

        self.plotter.render()

        self.log(f"\n📷 Camera focused on {structure_name}")
        self.log(f"   Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        self.log(f"   Size: {size:.1f} mm")
        self.log(f"   Camera distance: {camera_distance:.1f} mm")
        self.log("=" * 50)
        self.log("💡 Selected structure is now OPAQUE and ZOOMED")
        self.log("💡 Other structures are TRANSPARENT (faded)\n")

        return True

    def reset_focus(self):
        """
        Reset all structures to normal opacity and reset camera
        """
        self.log("\n" + "=" * 50)
        self.log("🔄 RESETTING FOCUS - Showing all structures equally")
        self.log("=" * 50)
        self.current_focus = None

        # Reset all opacities to normal
        for surf in self.surfaces:
            name = surf['name']
            actor = self.name_to_actor.get(name)
            if actor:
                actor.GetProperty().SetOpacity(self.NORMAL_OPACITY)
                self.log(f"  ✓ {name}: opacity = {self.NORMAL_OPACITY}")

        # Reset camera to show entire heart
        self.plotter.reset_camera()
        self.plotter.view_isometric()
        self.plotter.render()

        self.log("=" * 50)
        self.log("✅ Focus reset complete - all structures visible equally")
        self.log("💡 All structures now have normal opacity\n")
        return True

    def get_current_focus(self):
        """
        Returns name of currently focused structure or None
        """
        return self.current_focus

    def is_focusing(self):
        """
        Returns True if currently focusing on a structure
        """
        return self.current_focus is not None