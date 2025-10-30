import vtk
from PyQt5 import QtWidgets, QtCore, QtGui
import pyvista as pv

# optional safety (avoid empty-mesh crashes)
pv.global_theme.allow_empty_mesh = True


class ClippingControlWindow(QtWidgets.QWidget):
    def __init__(self, plotter, meshes):
        super().__init__()
        self.setWindowTitle("✂️ Clipping Plane Controls")
        self.setGeometry(300, 200, 420, 380)
        self.setWindowFlags(
            QtCore.Qt.Window |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.CustomizeWindowHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.WindowCloseButtonHint
        )

        self.plotter = plotter
        self.meshes = meshes  # List of PyVista PolyData objects
        self.planes = {}
        self.plane_actors = {}  # Store plane visualization actors
        self.original_meshes = meshes.copy()  # Store originals for restoration
        self.clipped_actors = []  # Track clipped mesh actors
        self.original_actors = []  # Track original actors to hide/show them

        # Pre-compute and cache clipped versions for performance
        self.vtk_clippers = {}  # Cache VTK clipper objects per axis

        # Store references to original actors from the plotter
        self.store_original_actors()

        # Validate meshes
        print(
            f"📦 Clipping Controls initialized with {len(self.meshes)} meshes")
        for i, mesh in enumerate(self.meshes):
            if mesh is not None:
                print(
                    f"   Mesh {i}: {mesh.n_points} points, {mesh.n_cells} cells")

        self.apply_dark_theme()
        self.build_ui()

        # Prevent the window from closing automatically when focus changes
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)

    def store_original_actors(self):
        """Store references to original actors so we can hide/show them"""
        try:
            # Get all current actors in the renderer
            actor_collection = self.plotter.renderer.GetActors()
            actor_collection.InitTraversal()

            for i in range(actor_collection.GetNumberOfItems()):
                actor = actor_collection.GetNextActor()
                if actor is not None:
                    self.original_actors.append(actor)

            print(f"📦 Stored {len(self.original_actors)} original actors")
        except Exception as e:
            print(f"⚠️ Warning: Could not store original actors: {e}")

    # ------------------------ UI ------------------------

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f1923, stop:1 #16213e);
                color: #ffffff;
                font-family: 'Segoe UI';
                font-size: 13px;
            }
            QCheckBox {
                spacing: 8px;
                font-weight: bold;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #2a3f5f;
                background-color: transparent;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #00bfa5;
                background-color: #00bfa5;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #2a3f5f;
                height: 6px;
                background: #2a3f5f;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #10b981;
                border: 1px solid #0d7055;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QLabel {
                color: #ccc;
            }
            QGroupBox {
                border: 1px solid #2a3f5f;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00bfa5;
            }
        """)

    def build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)

        title = QtWidgets.QLabel("3D Clipping Planes")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setFont(QtGui.QFont("Segoe UI", 14, QtGui.QFont.Bold))
        title.setStyleSheet("color:#10b981; margin-bottom:8px;")
        layout.addWidget(title)

        info = QtWidgets.QLabel(
            f"📦 Working with {len(self.meshes)} structures")
        info.setStyleSheet("color:#888; font-size:11px; font-style:italic;")
        info.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(info)

        # --- Plane Toggles ---
        plane_box = QtWidgets.QGroupBox("Enable Planes")
        vbox_planes = QtWidgets.QVBoxLayout(plane_box)

        for plane_name in ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]:
            cb = QtWidgets.QCheckBox(f"{plane_name}")
            cb.stateChanged.connect(
                lambda state, p=plane_name: self.toggle_plane(p, state))
            vbox_planes.addWidget(cb)
        layout.addWidget(plane_box)

        # --- Sliders ---
        slider_box = QtWidgets.QGroupBox("Move Planes")
        vbox_sliders = QtWidgets.QVBoxLayout(slider_box)

        self.sliders = {}
        for axis in ["x", "y", "z"]:
            label = QtWidgets.QLabel(f"{axis.upper()} position:")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(50)
            slider.setTracking(True)  # Enable real-time tracking
            slider.valueChanged.connect(
                lambda val, a=axis: self.move_plane(a, val))
            vbox_sliders.addWidget(label)
            vbox_sliders.addWidget(slider)
            self.sliders[axis] = slider
        layout.addWidget(slider_box)

        # --- Plane Visibility Toggle ---
        visibility_box = QtWidgets.QGroupBox("Plane Visibility")
        visibility_layout = QtWidgets.QVBoxLayout(visibility_box)

        self.show_planes_checkbox = QtWidgets.QCheckBox("Show Clipping Planes")
        self.show_planes_checkbox.setChecked(True)
        self.show_planes_checkbox.stateChanged.connect(
            self.toggle_plane_visibility)
        visibility_layout.addWidget(self.show_planes_checkbox)
        layout.addWidget(visibility_box)

        # --- Reset Button ---
        reset_btn = QtWidgets.QPushButton("🔄 Reset All")
        reset_btn.setMinimumHeight(40)
        reset_btn.setStyleSheet("""
            QPushButton {
                background: #2a3f5f;
                color: white;
                border: 2px solid #00bfa5;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #00bfa5;
                color: #0f1923;
            }
        """)
        reset_btn.clicked.connect(self.reset_all)
        layout.addWidget(reset_btn)

        layout.addStretch()

    # ------------------------ LOGIC ------------------------

    def toggle_plane(self, plane_name, state):
        """Enable/disable a clipping plane"""
        axis = plane_name.split("(")[1][0].lower()  # 'x', 'y', or 'z'

        if state == QtCore.Qt.Checked:
            # Create VTK plane
            plane = vtk.vtkPlane()
            if axis == "x":
                plane.SetNormal(1, 0, 0)
            elif axis == "y":
                plane.SetNormal(0, 1, 0)
            elif axis == "z":
                plane.SetNormal(0, 0, 1)

            self.planes[axis] = plane
            print(f"✅ Enabled {plane_name} clipping plane")

            # Initialize position and create visualization
            self.move_plane(axis, self.sliders[axis].value())
        else:
            # Remove plane
            if axis in self.planes:
                del self.planes[axis]
                print(f"❌ Disabled {plane_name} clipping plane")

                # Remove plane visualization
                self.remove_plane_visualization(axis)

                # Immediate update when disabling
                self.update_clipping_realtime()

    def create_plane_visualization(self, axis, origin, normal):
        """Create a visible plane mesh for visualization - OPTIMIZED"""
        # Calculate bounds once and cache
        if not hasattr(self, '_cached_bounds'):
            all_bounds = []
            for mesh in self.meshes:
                if mesh is not None and mesh.n_points > 0:
                    all_bounds.append(mesh.bounds)

            if not all_bounds:
                return

            import numpy as np
            bounds_array = np.array(all_bounds)
            self._cached_bounds = {
                'x_min': bounds_array[:, 0].min(),
                'x_max': bounds_array[:, 1].max(),
                'y_min': bounds_array[:, 2].min(),
                'y_max': bounds_array[:, 3].max(),
                'z_min': bounds_array[:, 4].min(),
                'z_max': bounds_array[:, 5].max(),
            }

        bounds = self._cached_bounds

        # Determine plane size based on orientation
        if axis == "x":
            i_size = (bounds['y_max'] - bounds['y_min']) * 1.3
            j_size = (bounds['z_max'] - bounds['z_min']) * 1.3
            color = "#FF6B6B"  # Red for X
        elif axis == "y":
            i_size = (bounds['x_max'] - bounds['x_min']) * 1.3
            j_size = (bounds['z_max'] - bounds['z_min']) * 1.3
            color = "#4ECDC4"  # Cyan for Y
        else:  # z
            i_size = (bounds['x_max'] - bounds['x_min']) * 1.3
            j_size = (bounds['y_max'] - bounds['y_min']) * 1.3
            color = "#95E1D3"  # Green for Z

        # Create plane mesh
        plane_mesh = pv.Plane(
            center=origin,
            direction=normal,
            i_size=i_size,
            j_size=j_size,
            i_resolution=2,  # Low resolution for performance
            j_resolution=2
        )

        # Remove old plane actor if exists
        self.remove_plane_visualization(axis)

        # Add new plane with semi-transparency and grid
        if self.show_planes_checkbox.isChecked():
            actor = self.plotter.add_mesh(
                plane_mesh,
                color=color,
                opacity=0.3,
                show_edges=True,
                edge_color='yellow',
                line_width=2,
                name=f'plane_{axis}',
                render=False  # Don't render immediately
            )
            self.plane_actors[axis] = actor

    def remove_plane_visualization(self, axis):
        """Remove the visual plane mesh"""
        if axis in self.plane_actors:
            try:
                self.plotter.remove_actor(
                    self.plane_actors[axis], render=False)
            except:
                pass
            del self.plane_actors[axis]

    def toggle_plane_visibility(self, state):
        """Show/hide all plane visualizations"""
        if state == QtCore.Qt.Checked:
            # Recreate all plane visualizations
            for axis in self.planes.keys():
                self.update_plane_visualization(axis)
            print("👁️ Clipping planes visible")
        else:
            # Hide all planes
            for axis in list(self.plane_actors.keys()):
                self.remove_plane_visualization(axis)
            print("👻 Clipping planes hidden")

        self.plotter.render()

    def update_plane_visualization(self, axis):
        """Update the visual representation of a plane"""
        if axis not in self.planes:
            return

        plane = self.planes[axis]
        origin = plane.GetOrigin()
        normal = plane.GetNormal()

        self.create_plane_visualization(axis, origin, normal)

    def move_plane(self, axis, value):
        """Move plane along its normal direction - REAL-TIME OPTIMIZED"""
        if axis not in self.planes or not self.meshes:
            return

        # Use cached bounds
        if not hasattr(self, '_cached_bounds'):
            all_bounds = []
            for mesh in self.meshes:
                if mesh is not None and mesh.n_points > 0:
                    all_bounds.append(mesh.bounds)

            if not all_bounds:
                return

            import numpy as np
            bounds_array = np.array(all_bounds)
            self._cached_bounds = {
                'x_min': bounds_array[:, 0].min(),
                'x_max': bounds_array[:, 1].max(),
                'y_min': bounds_array[:, 2].min(),
                'y_max': bounds_array[:, 3].max(),
                'z_min': bounds_array[:, 4].min(),
                'z_max': bounds_array[:, 5].max(),
            }

        bounds = self._cached_bounds
        center = [
            (bounds['x_min'] + bounds['x_max']) / 2,
            (bounds['y_min'] + bounds['y_max']) / 2,
            (bounds['z_min'] + bounds['z_max']) / 2,
        ]

        val = value / 100.0

        # Set plane origin based on axis
        if axis == "x":
            pos = bounds['x_min'] + val * (bounds['x_max'] - bounds['x_min'])
            origin = [pos, center[1], center[2]]
            normal = [1, 0, 0]
            self.planes[axis].SetOrigin(*origin)
        elif axis == "y":
            pos = bounds['y_min'] + val * (bounds['y_max'] - bounds['y_min'])
            origin = [center[0], pos, center[2]]
            normal = [0, 1, 0]
            self.planes[axis].SetOrigin(*origin)
        elif axis == "z":
            pos = bounds['z_min'] + val * (bounds['z_max'] - bounds['z_min'])
            origin = [center[0], center[1], pos]
            normal = [0, 0, 1]
            self.planes[axis].SetOrigin(*origin)

        # Update plane visualization
        self.create_plane_visualization(axis, origin, normal)

        # Real-time clipping update - NO DEBOUNCING
        self.update_clipping_realtime()

    def hide_original_actors(self):
        """Hide the original actors (red heart meshes)"""
        for actor in self.original_actors:
            actor.SetVisibility(False)

    def show_original_actors(self):
        """Show the original actors (red heart meshes)"""
        for actor in self.original_actors:
            actor.SetVisibility(True)

    def update_clipping_realtime(self):
        """Apply clipping in real-time - HIGHLY OPTIMIZED"""
        clip_functions = list(self.planes.values())

        # Clear only the clipped actors (batch operation)
        for actor in self.clipped_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except:
                pass
        self.clipped_actors = []

        # No planes active → restore original visibility
        if not clip_functions:
            self.show_original_actors()
            self.plotter.render()
            return

        # Hide original actors when clipping is active
        self.hide_original_actors()

        # Apply clipping to each mesh with minimal overhead
        for original_mesh in self.original_meshes:
            if original_mesh is None or original_mesh.n_points == 0:
                continue

            clipped = original_mesh

            # Apply each active plane sequentially
            for plane in clip_functions:
                clipper = vtk.vtkClipPolyData()
                clipper.SetInputData(clipped)
                clipper.SetClipFunction(plane)
                clipper.GenerateClippedOutputOff()  # Don't generate inverse
                clipper.Update()
                clipped = clipper.GetOutput()

                # Early exit if empty
                if clipped.GetNumberOfPoints() == 0:
                    break

            # Only add if result is not empty
            if clipped is not None and clipped.GetNumberOfPoints() > 0:
                actor = self.plotter.add_mesh(
                    clipped,
                    color="#9ca3af",
                    opacity=1.0,
                    smooth_shading=True,
                    lighting=True,
                    render=False  # Batch rendering
                )
                if actor is not None:
                    self.clipped_actors.append(actor)

        # Single efficient render call
        self.plotter.render()

    def reset_all(self):
        """Reset all planes and sliders"""
        # Clear planes
        self.planes.clear()

        # Remove all plane visualizations
        for axis in list(self.plane_actors.keys()):
            self.remove_plane_visualization(axis)

        # Clear clipped actors
        for actor in self.clipped_actors:
            try:
                self.plotter.remove_actor(actor, render=False)
            except:
                pass
        self.clipped_actors = []

        # Show original actors
        self.show_original_actors()

        # Reset sliders
        for slider in self.sliders.values():
            slider.setValue(50)

        # Clear cached bounds
        if hasattr(self, '_cached_bounds'):
            del self._cached_bounds

        self.plotter.render()
        print("🔄 All clipping planes reset")
