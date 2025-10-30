import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyvista as pv


class NIfTIClippingDialog(QtWidgets.QDialog):
    """
    Dialog for NIfTI volume clipping with proper viewer integration
    Live clipping plane + live 2D slice preview + (NEW) 3D MPR slice planes
    """

    def __init__(self, parent, plotter, volume_data, current_surfaces, log_callback):
        super().__init__(parent)

        self.parent_widget = parent
        self.plotter = plotter
        self.volume_data = volume_data

        # keep original meshes so we can re-clip them as slider moves
        self.current_surfaces = current_surfaces
        self.original_meshes = []
        for surf in self.current_surfaces:
            if 'mesh' in surf and surf['mesh'] is not None:
                self.original_meshes.append(surf['mesh'].copy())
            else:
                self.original_meshes.append(None)

        self.log_message = log_callback

        # --- State
        self.current_plane_actor = None
        self.current_plane_type = None
        self.current_plane_position = 50
        self.clipped_actors = []

        # (NEW) Remember per-plane positions (so the 3D MPR planes can be independent)
        self.per_plane_percent = {'axial': 50, 'sagittal': 50, 'coronal': 50}

        # (NEW) 3D MPR slice plane actors (textured planes)
        self.slice3d_actors = {'axial': None, 'sagittal': None, 'coronal': None}
        self.showing_slices_3d = False

        # 2D viewer target
        self.slice_viewer_frame = None
        self.slice_image_label = None
        self.label_2d = None

        self.find_viewer_components()
        self.init_ui()

        # live updates
        self.plane_combo.currentIndexChanged.connect(self._on_plane_or_slider_changed)
        self.position_slider.valueChanged.connect(self._on_plane_or_slider_changed)

        self.apply_clipping_live()

    def find_viewer_components(self):
        try:
            for child in self.parent_widget.findChildren(QtWidgets.QLabel):
                if 'slice' in child.objectName().lower() or 'slice' in child.text().lower():
                    self.slice_image_label = child
                    self.log_message("✅ Found slice image label")
                    break
            self.create_embedded_viewer = (self.slice_image_label is None)
        except Exception as e:
            self.log_message(f"⚠️ Viewer search error: {e}")
            self.create_embedded_viewer = True

    def init_ui(self):
        self.setWindowTitle("✂️ NIfTI Volume Clipping")
        self.setModal(False)
        self.setMinimumWidth(600)
        self.setMinimumHeight(700 if getattr(self, 'create_embedded_viewer', True) else 430)

        self.setStyleSheet("""
            QDialog { background-color: #1f2833; }
            QLabel { color: #e0e0e0; font-size: 13px; }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0f4c75, stop:1 #1b262c);
                color: white; border: 2px solid #00d4ff; border-radius: 8px; padding: 10px 20px;
                font-size: 12px; font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #00d4ff, stop:1 #0f4c75);
            }
            QGroupBox { background-color: #2a2a3e; border: 2px solid #00d4ff; border-radius: 8px;
                        padding: 15px; margin-top: 10px; font-weight: bold; color: #00d4ff; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15); layout.setContentsMargins(20, 20, 20, 20)

        title = QtWidgets.QLabel("✂️ NIfTI Volume Clipping: Multi-Planar Reconstruction")
        title.setStyleSheet("color: #00d4ff; font-size: 16px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        desc = QtWidgets.QLabel(
            "Move the slider to slice through the volume.\n"
            "3D is clipped in real time, 2D slice is shown below.\n"
            "Use “Visualize Slices” to place the three MPR planes in 3D."
        )
        desc.setStyleSheet("color: #00ff00; font-size: 11px; font-style: italic;")
        desc.setWordWrap(True); desc.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(desc)

        plane_group = QtWidgets.QGroupBox("Plane Orientation")
        plane_layout = QtWidgets.QVBoxLayout()
        self.plane_combo = QtWidgets.QComboBox()
        self.plane_combo.addItem("Axial (Top → Bottom)")
        self.plane_combo.addItem("Sagittal (Right → Left)")
        self.plane_combo.addItem("Coronal (Front → Back)")
        self.plane_combo.setMinimumHeight(40)
        plane_layout.addWidget(self.plane_combo)
        plane_group.setLayout(plane_layout)
        layout.addWidget(plane_group)

        position_group = QtWidgets.QGroupBox("Position")
        position_layout = QtWidgets.QVBoxLayout()
        self.position_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.position_slider.setMinimum(0); self.position_slider.setMaximum(100)
        self.position_slider.setValue(50); self.position_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.position_slider.setTickInterval(10)
        position_layout.addWidget(self.position_slider)
        self.position_value_label = QtWidgets.QLabel("Position: 50%")
        self.position_value_label.setStyleSheet("color: #00d4ff; font-size: 11px; font-weight: bold;")
        self.position_value_label.setAlignment(QtCore.Qt.AlignCenter)
        position_layout.addWidget(self.position_value_label)
        position_group.setLayout(position_layout)
        layout.addWidget(position_group)

        if getattr(self, 'create_embedded_viewer', True):
            viewer_group = QtWidgets.QGroupBox("📊 2D Slice View")
            viewer_layout = QtWidgets.QVBoxLayout()
            self.embedded_slice_label = QtWidgets.QLabel()
            self.embedded_slice_label.setAlignment(QtCore.Qt.AlignCenter)
            self.embedded_slice_label.setMinimumHeight(250)
            self.embedded_slice_label.setStyleSheet("""
                QLabel { background-color: #000000; border: 2px solid #ff00ff; padding: 5px; }
            """)
            self.embedded_slice_label.setText("No slice yet")
            viewer_layout.addWidget(self.embedded_slice_label)
            viewer_group.setLayout(viewer_layout)
            layout.addWidget(viewer_group)
            if self.slice_image_label is None:
                self.slice_image_label = self.embedded_slice_label

        # --- Buttons
        btn_layout_top = QtWidgets.QHBoxLayout()

        btn_apply = QtWidgets.QPushButton("✂️ Apply Clipping")
        btn_apply.setMinimumHeight(46)
        btn_apply.clicked.connect(self.apply_clipping_live)
        btn_layout_top.addWidget(btn_apply)

        # (NEW) Button: place three orthogonal slice planes (textured) in 3D
        self.visualize_slices_button = QtWidgets.QPushButton("📐 Visualize Slices (3D MPR)")
        self.visualize_slices_button.setMinimumHeight(46)
        self.visualize_slices_button.clicked.connect(self.visualize_slices_3d)
        btn_layout_top.addWidget(self.visualize_slices_button)

        layout.addLayout(btn_layout_top)

        btn_layout_bottom = QtWidgets.QHBoxLayout()
        btn_clear = QtWidgets.QPushButton("🔄 Clear Plane")
        btn_clear.setMinimumHeight(46)
        btn_clear.clicked.connect(self.clear_clipping_plane)
        btn_layout_bottom.addWidget(btn_clear)

        btn_close = QtWidgets.QPushButton("✖ Close")
        btn_close.setMinimumHeight(46)
        btn_close.clicked.connect(self.close)
        btn_layout_bottom.addWidget(btn_close)

        layout.addLayout(btn_layout_bottom)

    # ----------------------------------------------
    # Live clipping + 2D label (unchanged behavior)
    # ----------------------------------------------
    def _on_plane_or_slider_changed(self, *args):
        self.update_position_label(self.position_slider.value())
        self.apply_clipping_live()
        # (NEW) if MPR planes are on, keep them in sync too
        if self.showing_slices_3d:
            self._update_slices_3d_images()

    def update_position_label(self, value):
        self.position_value_label.setText(f"Position: {value}%")

    def _get_plane_type_from_ui(self):
        t = self.plane_combo.currentText()
        if "Axial" in t: return "axial", "Axial"
        if "Sagittal" in t: return "sagittal", "Sagittal"
        if "Coronal" in t: return "coronal", "Coronal"
        return "axial", "Axial"

    def apply_clipping_live(self):
        plane_type, plane_name = self._get_plane_type_from_ui()
        position_percent = self.position_slider.value()

        # remember per-plane position
        self.per_plane_percent[plane_type] = position_percent

        self.log_message(f"\n✂️ Live {plane_name} clipping at {position_percent}%")
        self.current_plane_type = plane_type
        self.current_plane_position = position_percent

        clip_center, clip_normal = self.update_cutting_plane_actor(plane_type, position_percent)
        self.update_clipped_meshes(clip_center, clip_normal)
        self.extract_and_display_slice(plane_type, position_percent)

    def update_cutting_plane_actor(self, plane_type, position_percent):
        if self.current_plane_actor is not None:
            try: self.plotter.remove_actor(self.current_plane_actor)
            except: pass
            self.current_plane_actor = None

        if not self.current_surfaces:
            return ([0,0,0],[0,0,1])

        # bounds of all meshes
        all_bounds = [s['mesh'].bounds for s in self.current_surfaces if 'mesh' in s and s['mesh'] is not None]
        b = np.array(all_bounds)
        x_min, x_max = b[:,0].min(), b[:,1].max()
        y_min, y_max = b[:,2].min(), b[:,3].max()
        z_min, z_max = b[:,4].min(), b[:,5].max()

        if plane_type == "axial":
            z_pos = z_min + (z_max - z_min) * (position_percent/100.0)
            center = [(x_min+x_max)/2, (y_min+y_max)/2, z_pos]
            normal = [0,0,1]; i_size = (x_max-x_min)*1.2; j_size = (y_max-y_min)*1.2
        elif plane_type == "sagittal":
            x_pos = x_min + (x_max - x_min) * (position_percent/100.0)
            center = [x_pos, (y_min+y_max)/2, (z_min+z_max)/2]
            normal = [1,0,0]; i_size = (y_max-y_min)*1.2; j_size = (z_max-z_min)*1.2
        else:
            y_pos = y_min + (y_max - y_min) * (position_percent/100.0)
            center = [(x_min+x_max)/2, y_pos, (z_min+z_max)/2]
            normal = [0,1,0]; i_size = (x_max-x_min)*1.2; j_size = (z_max-z_min)*1.2

        plane = pv.Plane(center=center, direction=normal, i_size=i_size, j_size=j_size)
        self.current_plane_actor = self.plotter.add_mesh(
            plane, color='cyan', opacity=0.35, show_edges=True,
            edge_color='yellow', line_width=2, name='nifti_cutting_plane'
        )
        self.plotter.render()
        return center, normal

    def update_clipped_meshes(self, plane_center, plane_normal):
        for a in self.clipped_actors:
            try: self.plotter.remove_actor(a)
            except: pass
        self.clipped_actors = []

        if not self.current_surfaces: return

        for idx, surf in enumerate(self.current_surfaces):
            base_mesh = self.original_meshes[idx]
            if base_mesh is None: continue
            try:
                clipped = base_mesh.clip(normal=plane_normal, origin=plane_center, invert=False)
                if 'actor' in surf and surf['actor'] is not None:
                    try: self.plotter.remove_actor(surf['actor'])
                    except: pass
                    surf['actor'] = None

                color = surf.get('color', 'white')
                new_actor = self.plotter.add_mesh(clipped, color=color, opacity=surf.get('opacity',1.0),
                                                  name=f"clipped_{idx}")
                self.clipped_actors.append(new_actor)
                surf['actor'] = new_actor
            except Exception as e:
                self.log_message(f"⚠️ clipping error on surface {idx}: {e}")
        self.plotter.render()

    def extract_and_display_slice(self, plane_type, position_percent):
        if self.volume_data is None:
            self.log_message("⚠️ No volume data"); return
        shape = self.volume_data.shape  # (X,Y,Z)

        if plane_type == "axial":
            max_idx = shape[2]-1; i = int((position_percent/100.0)*max_idx); data = self.volume_data[:,:,i]
        elif plane_type == "sagittal":
            max_idx = shape[0]-1; i = int((position_percent/100.0)*max_idx); data = self.volume_data[i,:,:]
        else:
            max_idx = shape[1]-1; i = int((position_percent/100.0)*max_idx); data = self.volume_data[:,i,:]

        mn, mx = float(data.min()), float(data.max())
        img = ((data - mn)/(mx-mn)*255).astype(np.uint8) if mx>mn else np.zeros_like(data, np.uint8)
        img = np.rot90(img); img = np.ascontiguousarray(img)

        h,w = img.shape
        qimg = QtGui.QImage(img.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        if self.slice_image_label:
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self.slice_image_label.width()-20, self.slice_image_label.height()-20,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.slice_image_label.setPixmap(pix)
            self.log_message(f"✅ 2D slice displayed: {plane_type} @ {position_percent}%")

    # ------------------------------------------------------
    # (NEW) 3D MPR: orthogonal textured planes inside PyVista
    # ------------------------------------------------------
    def visualize_slices_3d(self):
        """Toggle 3D slice planes (axial/sagittal/coronal) in the 3D scene."""
        if self.volume_data is None:
            QtWidgets.QMessageBox.warning(self, "No NIfTI Loaded", "Please load a NIfTI file first.")
            return

        # --- Toggle behavior
        if self.showing_slices_3d:
            # Remove all existing 3D slice planes
            for key, act in self.slice3d_actors.items():
                if act is not None:
                    try:
                        self.plotter.remove_actor(act)
                    except Exception:
                        pass
                    self.slice3d_actors[key] = None
            self.showing_slices_3d = False
            self.visualize_slices_button.setText("📐 Visualize Slices (3D MPR)")
            self.log_message("📉 3D MPR planes hidden")
        else:
            # Create them fresh
            self.showing_slices_3d = True
            self._update_slices_3d_images()
            self.visualize_slices_button.setText("❌ Hide Slices (3D MPR)")
            self.log_message("📐 3D MPR planes displayed")

    def _build_texture_from_slice(self, slice_2d):
        """Convert a 2D numpy slice to a PyVista texture (RGB)."""
        smin = float(slice_2d.min())
        smax = float(slice_2d.max())
        if smax <= smin:
            gray = np.zeros_like(slice_2d, dtype=np.uint8)
        else:
            gray = ((slice_2d - smin) / (smax - smin) * 255.0).astype(np.uint8)

        # ensure contiguous RGB uint8
        rgb = np.dstack((gray, gray, gray))
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        tex = pv.Texture(rgb)  # texture, not per-vertex RGB scalars
        # You can tweak interpolation if you want:
        # tex.interpolate = False

        # return texture and dimensions (width=x, height=y)
        return tex, rgb.shape[1], rgb.shape[0]

    def _update_slices_3d_images(self):
        """(Re)create planes with textures and place them at stored percents."""
        if self.volume_data is None: return

        # If we have no surface bounds, synthesize from volume dimensions
        # Create a virtual bounding box: x=[0,nx], y=[0,ny], z=[0,nz]
        nx, ny, nz = self.volume_data.shape
        x_min, x_max = 0.0, float(nx)
        y_min, y_max = 0.0, float(ny)
        z_min, z_max = 0.0, float(nz)

        # Helper to compute world coords from percent
        def pos_from_percent(axis, p):
            p = np.clip(p, 0, 100)/100.0
            if axis == 'x': return x_min + (x_max-x_min)*p
            if axis == 'y': return y_min + (y_max-y_min)*p
            return z_min + (z_max-z_min)*p

        # --- Axial (XY plane @ z)
        ia = int((self.per_plane_percent['axial']/100.0) * (nz-1))
        axial2d = self.volume_data[:, :, ia]
        tex_ax, w_ax, h_ax = self._build_texture_from_slice(axial2d)
        z = pos_from_percent('z', self.per_plane_percent['axial'])
        # plane extent in world units (x_max-x_min by y_max-y_min)
        plane_ax = pv.Plane(center=[(x_max+x_min)/2, (y_max+y_min)/2, z],
                            direction=[0,0,1],
                            i_size=(x_max-x_min), j_size=(y_max-y_min))
        self._add_or_replace_slice_actor('axial', plane_ax, tex_ax)

        # --- Sagittal (YZ plane @ x)
        isg = int((self.per_plane_percent['sagittal']/100.0) * (nx-1))
        sag2d = self.volume_data[isg, :, :]
        tex_sg, _, _ = self._build_texture_from_slice(sag2d)
        x = pos_from_percent('x', self.per_plane_percent['sagittal'])
        plane_sg = pv.Plane(center=[x, (y_max+y_min)/2, (z_max+z_min)/2],
                            direction=[1,0,0],
                            i_size=(y_max-y_min), j_size=(z_max-z_min))
        self._add_or_replace_slice_actor('sagittal', plane_sg, tex_sg)

        # --- Coronal (XZ plane @ y)
        ic = int((self.per_plane_percent['coronal']/100.0) * (ny-1))
        cor2d = self.volume_data[:, ic, :]
        tex_co, _, _ = self._build_texture_from_slice(cor2d)
        y = pos_from_percent('y', self.per_plane_percent['coronal'])
        plane_co = pv.Plane(center=[(x_max+x_min)/2, y, (z_max+z_min)/2],
                            direction=[0,1,0],
                            i_size=(x_max-x_min), j_size=(z_max-z_min))
        self._add_or_replace_slice_actor('coronal', plane_co, tex_co)

        self.plotter.render()

    def _add_or_replace_slice_actor(self, name, plane_mesh, texture):
        """Remove previous actor for a given plane name and add a new textured one."""
        # remove old
        old = self.slice3d_actors.get(name)
        if old is not None:
            try:
                self.plotter.remove_actor(old)
            except:
                pass
            self.slice3d_actors[name] = None

        # IMPORTANT: do NOT pass rgb=True here (that is for scalar RGB arrays)
        actor = self.plotter.add_mesh(
            plane_mesh,
            name=f"mpr_{name}",
            texture=texture,  # a pv.Texture
            smooth_shading=False,
            opacity=1.0,
            show_edges=False
            # no 'scalars', no 'rgb'
        )
        self.slice3d_actors[name] = actor

    # ------------------------------------------------------

    def clear_clipping_plane(self):
        # remove plane actor
        if self.current_plane_actor is not None:
            try: self.plotter.remove_actor(self.current_plane_actor)
            except: pass
            self.current_plane_actor = None

        # remove clipped actors
        for a in self.clipped_actors:
            try: self.plotter.remove_actor(a)
            except: pass
        self.clipped_actors = []

        # (NEW) remove the three MPR slice planes if shown
        for key, act in self.slice3d_actors.items():
            if act is not None:
                try: self.plotter.remove_actor(act)
                except: pass
                self.slice3d_actors[key] = None
        self.showing_slices_3d = False

        # restore originals
        for idx, surf in enumerate(self.current_surfaces):
            base_mesh = self.original_meshes[idx]
            if base_mesh is None: continue
            color = surf.get('color', 'white')
            restored = self.plotter.add_mesh(base_mesh, color=color, opacity=surf.get('opacity',1.0),
                                             name=f"restored_{idx}")
            surf['actor'] = restored

        self.plotter.render()

        if self.slice_image_label:
            self.slice_image_label.clear()
            self.slice_image_label.setText("No slice")
        self.log_message("🔄 Clipping + 3D MPR cleared")

    def closeEvent(self, event):
        self.clear_clipping_plane()
        event.accept()
