from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
import os
import numpy as np
import nibabel as nib
from PyQt5 import QtWidgets, QtCore, QtGui
import pyvista as pv
from pyvistaqt import QtInteractor
from skimage import measure

# IMPORT BACKEND FEATURES
from feature_show_anatomy import build_heart_surfaces_from_seg
from feature_focus_navigation import FocusNavigationController
from flythrough_fixed import FlythroughController
from heart_fixed import HeartPumpController
from curved_mpr import CurvedMPRController, CurvedMPRDialog
from clipping_controls import ClippingControlWindow
from mpr import NIfTIClippingDialog
from manual_flythrough_FIXED import ManualFlythroughController
from anatomy_transparency_module import AnatomyTransparencyController
from selective_removal_module import SelectiveRemovalController
from custom_order_flythrough import CustomOrderFlythroughController

import matplotlib

matplotlib.use('Qt5Agg')


# ==================== BRAIN ANIMATION CONTROLLER ====================
class BrainAnimationController:
    """Controller for neural brain signal animation"""

    # Colors
    BASE_BRAIN = np.array([0.3, 0.35, 0.4])
    BASE_BLOOD = np.array([0.3, 0.1, 0.1])
    SIGNAL_COLOR = np.array([0.0, 0.8, 1.0])
    SIGNAL_PEAK = np.array([1.0, 1.0, 1.0])

    # Neural pathways
    PATHS = {
        "thinking": ["brainstem", "thalamus", "frontal", "motor"],
        "seeing": ["occipital", "temporal", "frontal"],
        "hearing": ["temporal", "parietal", "frontal"]
    }

    KEYWORDS = {
        "brainstem": ["brainstem", "pons", "medulla"],
        "thalamus": ["thalamus"],
        "frontal": ["frontal"],
        "motor": ["motor", "precentral"],
        "temporal": ["temporal"],
        "occipital": ["occipital"],
        "parietal": ["parietal"],
        "blood": ["artery", "vein", "sinus", "vascular"],
    }

    def __init__(self, plotter, surfaces, console_log=None):
        self.plotter = plotter
        self.surfaces = surfaces
        self.log = console_log or print

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step_animation)

        self.animating = False
        self.pos = 0.0
        self.path = []

        # Organize surfaces by brain region
        self.by_bucket = {}
        for surf in self.surfaces:
            bucket = self._bucket_for(surf['name'])
            if bucket not in self.by_bucket:
                self.by_bucket[bucket] = []
            self.by_bucket[bucket].append(surf)

            # Store original color
            base = self.BASE_BLOOD.copy() if bucket == "blood" else self.BASE_BRAIN.copy()
            surf['base_color'] = base

        self.log("🧠 Brain animation controller initialized")

    def _bucket_for(self, name):
        """Classify structure into brain region"""
        n = name.lower()
        for k, words in self.KEYWORDS.items():
            if any(w in n for w in words):
                return k
        return "other"

    def _mix_colors(self, c1, c2, t):
        """Linear interpolation between colors"""
        return (1 - t) * c1 + t * c2

    def _smoothstep(self, x):
        """Smooth interpolation curve"""
        x = np.clip(x, 0, 1)
        return 3 * x ** 2 - 2 * x ** 3

    def start_animation(self, pathway_name):
        """Start neural signal animation along pathway"""
        if self.animating:
            self.log("⚠️ Animation already running")
            return

        if pathway_name not in self.PATHS:
            self.log(f"❌ Unknown pathway: {pathway_name}")
            return

        # Build path from available regions
        self.path = [b for b in self.PATHS[pathway_name]
                     if b in self.by_bucket]

        if not self.path:
            self.log(f"❌ No brain regions found for {pathway_name}")
            return

        self.pos = -0.5
        self.animating = True
        self.timer.start(30)

        self.log(f"🧠 Neural signal: {pathway_name.upper()} pathway")
        self.log(f"   Path: {' → '.join(self.path)}")

    def step_animation(self):
        """Update animation frame"""
        self.pos += 0.05

        for i, region in enumerate(self.path):
            d = i - self.pos

            if d < -1.2:
                color = self.by_bucket[region][0]['base_color']
            elif -0.2 <= d <= 0.2:
                color = self._mix_colors(
                    self.SIGNAL_COLOR,
                    self.SIGNAL_PEAK,
                    self._smoothstep(1 - abs(d) * 5)
                )
            elif -1.0 <= d < -0.2:
                fade = self._smoothstep(1 - abs(d))
                color = self._mix_colors(
                    self.by_bucket[region][0]['base_color'],
                    self.SIGNAL_COLOR,
                    fade
                )
            else:
                color = self.by_bucket[region][0]['base_color']

            for surf in self.by_bucket[region]:
                if 'actor' in surf and surf['actor'] is not None:
                    surf['actor'].GetProperty().SetColor(*color)

        self.plotter.render()

        if self.pos > len(self.path):
            self.stop_animation()
            QtCore.QTimer.singleShot(1000, self.fade_back_to_base)

    def stop_animation(self):
        """Stop animation"""
        self.timer.stop()
        self.animating = False
        self.log("⏹️ Animation stopped")

    def fade_back_to_base(self):
        """Reset all colors to base"""
        for surf in self.surfaces:
            if 'base_color' in surf and 'actor' in surf and surf['actor'] is not None:
                surf['actor'].GetProperty().SetColor(*surf['base_color'])
        self.plotter.render()
        self.log("✅ Brain reset to base colors")


# ==================== MAIN WINDOW ====================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biomedical Engineering Visualization Platform")
        self.setMinimumSize(1600, 900)

        # Global dark style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QTabWidget::pane {
                border: 2px solid #16213e;
                border-radius: 8px;
                background-color: #16213e;
            }
            QTabBar::tab {
                background-color: #0f3460;
                color: #e0e0e0;
                padding: 12px 24px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #16213e;
                border: 2px solid #00d4ff;
            }
            QTabBar::tab:selected {
                background-color: #16213e;
                border: 2px solid #00d4ff;
                border-bottom: none;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #0f4c75, stop:1 #1b262c);
                color: white;
                border: 2px solid #00d4ff;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #00d4ff, stop:1 #0f4c75);
                border: 2px solid #00fff5;
            }
            QPushButton:pressed {
                background-color: #053742;
            }
            QPushButton:disabled {
                background-color: #2a2a3a;
                color: #666666;
                border: 2px solid #444444;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 11px;
            }
            QTextEdit {
                background-color: #0a0a0a;
                color: #00ff00;
                border: 2px solid #0f4c75;
                border-radius: 8px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
            }
            QComboBox {
                background-color: #1f2833;
                color: #e0e0e0;
                border: 2px solid #00d4ff;
                border-radius: 6px;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QComboBox:hover {
                border: 2px solid #00fff5;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 8px solid #00d4ff;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #1f2833;
                color: #e0e0e0;
                selection-background-color: #0f4c75;
                border: 2px solid #00d4ff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #0f4c75;
                height: 8px;
                background: #1f2833;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00d4ff;
                border: 2px solid #00fff5;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #00fff5;
            }
        """)

        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Tabs
        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.cardiovascular_tab = SystemTab("Cardiovascular", has_moving=True)
        self.tab_widget.addTab(self.cardiovascular_tab, "Cardiovascular")

        self.nervous_tab = SystemTab("Nervous", has_moving=True)
        self.tab_widget.addTab(self.nervous_tab, "Nervous")

        self.musculoskeletal_tab = SystemTab(
            "Musculoskeletal", has_moving=False)
        self.tab_widget.addTab(self.musculoskeletal_tab, "Musculoskeletal")

        self.dental_tab = SystemTab("Dental / Mouth", has_moving=False)
        self.tab_widget.addTab(self.dental_tab, "Dental / Mouth")


class SystemTab(QtWidgets.QWidget):
    """Unified tab with all features including brain segmentation loading"""

    def __init__(self, system_name, has_moving=False):
        super().__init__()
        self.system_name = system_name
        self.has_moving = has_moving

        # Data
        self.volume_path = None
        self.seg_path = None
        self.model_folder_path = None
        self.current_surfaces = []
        self.data_mode = None
        self.anatomy_controller = None
        self.removal_controller = None

        # Volume
        self.volume_data = None
        self.volume_affine = None
        self.volume_header = None

        # Controllers
        self.focus_controller = None
        self.flythrough_controller = None
        self.manual_flythrough_controller = None
        self.custom_order_controller = None
        self.pump_controller = None
        self.curved_mpr_controller = None
        self.brain_controller = None

        # Windows
        self.clipping_widget = None
        self.nifti_clipping_dialog = None
        self.stored_opacities = {}

        # Layout
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 #00d4ff, stop:0.5 #ff00ff, stop:1 #00d4ff);
            }
        """)

        # LEFT
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(10, 10, 10, 10)

        self.upload_panel = self.create_upload_panel()
        left_layout.addWidget(self.upload_panel)

        self.feature_panel = self.create_feature_panel()
        left_layout.addWidget(self.feature_panel)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # RIGHT
        self.viewer_panel = self.create_viewer_panel()
        splitter.addWidget(self.viewer_panel)

        splitter.setSizes([500, 1100])
        main_layout.addWidget(splitter)

    def create_upload_panel(self):
        panel = QtWidgets.QGroupBox()
        panel.setStyleSheet("""
            QGroupBox {
                background-color: #1f2833;
                border: 2px solid #00d4ff;
                border-left: 6px solid #00d4ff;
                border-radius: 10px;
                margin-top: 15px;
                padding: 15px;
                font-size: 13px;
                font-weight: bold;
                color: #00d4ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                color: #00d4ff;
            }
        """)
        panel.setTitle(f"📁 {self.system_name} Data Input")

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(15, 25, 15, 15)

        mode_label = QtWidgets.QLabel("Choose Data Source:")
        mode_label.setStyleSheet(
            "color: #00d4ff; font-size: 12px; font-weight: bold;")
        layout.addWidget(mode_label)

        seg_section = QtWidgets.QLabel("━━━ OPTION 1: Segmentation Files ━━━")
        seg_section.setStyleSheet(
            "color: #ff00ff; font-size: 11px; font-weight: bold; padding: 5px;")
        seg_section.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(seg_section)

        self.btn_volume = QtWidgets.QPushButton(
            "📤 Upload Volume\n(CT/MRI - .nii/.nii.gz)")
        self.btn_volume.setMinimumHeight(60)
        self.btn_volume.clicked.connect(self.browse_volume)
        layout.addWidget(self.btn_volume)

        self.lbl_volume_status = QtWidgets.QLabel("Status: No volume loaded")
        self.lbl_volume_status.setStyleSheet(
            "color: #999999; font-style: italic;")
        self.lbl_volume_status.setWordWrap(True)
        layout.addWidget(self.lbl_volume_status)

        self.btn_seg = QtWidgets.QPushButton(
            "📤 Upload Segmentation\n(.nii/.nii.gz file OR folder)")
        self.btn_seg.setMinimumHeight(60)
        self.btn_seg.clicked.connect(self.browse_seg)
        layout.addWidget(self.btn_seg)

        self.lbl_seg_status = QtWidgets.QLabel(
            "Status: No segmentation loaded")
        self.lbl_seg_status.setStyleSheet(
            "color: #999999; font-style: italic;")
        self.lbl_seg_status.setWordWrap(True)
        layout.addWidget(self.lbl_seg_status)

        brain_note = QtWidgets.QLabel(
            "💡 Can upload: Single file OR folder with multiple .nii files")
        brain_note.setStyleSheet(
            "color: #00d4ff; font-size: 10px; font-style: italic; padding: 5px;")
        brain_note.setWordWrap(True)
        layout.addWidget(brain_note)

        obj_section = QtWidgets.QLabel(
            "━━━ OPTION 2: 3D Models (Multi-OBJ) ━━━")
        obj_section.setStyleSheet(
            "color: #00ffaa; font-size: 11px; font-weight: bold; padding: 5px;")
        obj_section.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(obj_section)

        self.btn_model_folder = QtWidgets.QPushButton(
            "📤 Upload 3D Models Folder\n(Multiple .obj/.stl files)")
        self.btn_model_folder.setMinimumHeight(60)
        self.btn_model_folder.clicked.connect(self.browse_model_folder)
        layout.addWidget(self.btn_model_folder)

        self.lbl_model_status = QtWidgets.QLabel("Status: No models loaded")
        self.lbl_model_status.setStyleSheet(
            "color: #999999; font-style: italic;")
        self.lbl_model_status.setWordWrap(True)
        layout.addWidget(self.lbl_model_status)

        panel.setLayout(layout)
        return panel

    def create_feature_panel(self):
        panel = QtWidgets.QGroupBox()
        panel.setStyleSheet("""
            QGroupBox {
                background-color: #1f2833;
                border: 2px solid #ff00ff;
                border-left: 6px solid #ff00ff;
                border-radius: 10px;
                margin-top: 15px;
                padding: 15px;
                font-size: 13px;
                font-weight: bold;
                color: #ff00ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                color: #ff00ff;
            }
        """)
        panel.setTitle("🎨 Visualization Features")

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(15, 25, 15, 15)

        self.btn_anatomy = QtWidgets.QPushButton(
            "🫀 Show Anatomy\n(Color-coded Structure)")
        self.btn_anatomy.setMinimumHeight(65)
        self.btn_anatomy.clicked.connect(self.on_show_anatomy_clicked)
        layout.addWidget(self.btn_anatomy)

        self.btn_focus = QtWidgets.QPushButton(
            "🔍 Focus Navigation\n(Transparency + Zoom)")
        self.btn_focus.setMinimumHeight(65)
        self.btn_focus.setEnabled(False)
        self.btn_focus.clicked.connect(self.on_focus_navigation_clicked)
        layout.addWidget(self.btn_focus)

        self.btn_clipping = QtWidgets.QPushButton(
            "✂️ Clipping Plane\n(3D Cut)")
        self.btn_clipping.setMinimumHeight(65)
        self.btn_clipping.setEnabled(False)
        self.btn_clipping.clicked.connect(self.on_clipping_clicked)
        layout.addWidget(self.btn_clipping)

        self.btn_curved_mpr = QtWidgets.QPushButton(
            "🌊 Curved MPR\n(Curved Slice)")
        self.btn_curved_mpr.setMinimumHeight(65)
        self.btn_curved_mpr.setEnabled(False)
        self.btn_curved_mpr.clicked.connect(self.on_curved_mpr_clicked)
        layout.addWidget(self.btn_curved_mpr)

        self.btn_flythrough = QtWidgets.QPushButton(
            "🚀 Fly-through\n(Camera Path)")
        self.btn_flythrough.setMinimumHeight(65)
        self.btn_flythrough.setEnabled(False)
        self.btn_flythrough.clicked.connect(self.on_flythrough_clicked)
        layout.addWidget(self.btn_flythrough)

        if self.system_name == "Cardiovascular":
            moving_label = "💗 Moving Stuff\n(Heart Pumping)"
        elif self.system_name == "Nervous":
            moving_label = "🧠 Neural Activity\n(Brain Signals)"
        else:
            moving_label = "💫 Moving Stuff\n(Animation)"

        self.btn_moving = QtWidgets.QPushButton(moving_label)
        self.btn_moving.setMinimumHeight(65)
        self.btn_moving.setEnabled(False)
        if self.has_moving:
            self.btn_moving.clicked.connect(self.on_moving_stuff_clicked)
        layout.addWidget(self.btn_moving)

        self.btn_remove = QtWidgets.QPushButton(
            "🗑️ Remove Structures\n(Hide Ribs/Skull/etc)")
        self.btn_remove.setMinimumHeight(65)
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self.on_selective_removal_clicked)
        layout.addWidget(self.btn_remove)

        console_label = QtWidgets.QLabel("System Console:")
        console_label.setStyleSheet(
            "color: #00ff00; font-weight: bold; font-size: 11px; margin-top: 10px;")
        layout.addWidget(console_label)

        self.console = QtWidgets.QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(180)
        self.console.setMaximumHeight(220)
        self.console.setPlainText(f"{self.system_name} System ready.\n")
        layout.addWidget(self.console)

        panel.setLayout(layout)
        return panel

    def create_viewer_panel(self):
        panel = QtWidgets.QGroupBox()
        panel.setStyleSheet("""
            QGroupBox {
                background-color: #1f2833;
                border: 2px solid #00ffaa;
                border-left: 6px solid #00ffaa;
                border-radius: 10px;
                margin-top: 15px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #00ffaa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px 10px;
                color: #00ffaa;
            }
        """)
        panel.setTitle(f"🔬 {self.system_name} Visualization Window")

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(10, 25, 10, 10)
        main_layout.setSpacing(10)

        label_3d = QtWidgets.QLabel("🎯 3D View: Interactive Visualization")
        label_3d.setStyleSheet(
            "color: #00ffaa; font-size: 12px; font-weight: bold; padding: 3px;")
        label_3d.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(label_3d)

        self.plotter = QtInteractor(self)
        self.plotter.set_background('#1a1a1a', top='#2a2a3a')
        self.plotter.enable_anti_aliasing('msaa')
        main_layout.addWidget(self.plotter.interactor)

        self.slice_image_label = QtWidgets.QLabel()
        self.slice_image_label.setObjectName("slice_viewer_label")
        self.slice_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.slice_image_label.setMinimumHeight(150)
        self.slice_image_label.setMaximumHeight(200)
        self.slice_image_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 2px solid #ff00ff;
                padding: 5px;
                color: #666666;
            }
        """)
        self.slice_image_label.setText("2D Slice Viewer\n(NIfTI Clipping)")
        main_layout.addWidget(self.slice_image_label)

        hint = QtWidgets.QLabel(
            "💡 Mouse: Left=Rotate | Right=Pan | Scroll=Zoom")
        hint.setStyleSheet(
            "color: #999999; font-size: 10px; font-style: italic; padding: 5px;")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(hint)

        panel.setLayout(main_layout)
        return panel

    def log_message(self, msg):
        self.console.append(f"[{QtCore.QTime.currentTime().toString()}] {msg}")
        print(msg)

    def browse_volume(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Volume (CT/MRI)", "", "NIfTI (*.nii *.nii.gz);;All Files (*)"
        )
        if path:
            self.volume_path = path
            self.lbl_volume_status.setText(
                f"✅ Loaded: {os.path.basename(path)}")
            self.lbl_volume_status.setStyleSheet(
                "color: #00ff00; font-weight: bold;")
            self.log_message(f"Volume loaded: {os.path.basename(path)}")

            try:
                nii = nib.load(path)
                self.volume_data = nii.get_fdata()
                self.volume_affine = nii.affine
                self.volume_header = nii.header
                self.log_message(
                    f"✅ Volume data: shape {self.volume_data.shape}")
                self.btn_curved_mpr.setEnabled(True)
                self.log_message("🌊 Curved MPR enabled!")
            except Exception as e:
                self.log_message(f"⚠️ Volume load error: {e}")

    def browse_seg(self):
        """Browse for segmentation - supports both single file AND folder"""
        # Ask user what they want to upload
        choice_dialog = QtWidgets.QMessageBox(self)
        choice_dialog.setWindowTitle("Upload Segmentation")
        choice_dialog.setText("What would you like to upload?")
        choice_dialog.setIcon(QtWidgets.QMessageBox.Question)

        btn_file = choice_dialog.addButton(
            "📄 Single File (.nii/.nii.gz)", QtWidgets.QMessageBox.AcceptRole)
        btn_folder = choice_dialog.addButton(
            "📁 Folder (Multiple .nii files)", QtWidgets.QMessageBox.AcceptRole)
        btn_cancel = choice_dialog.addButton(
            "Cancel", QtWidgets.QMessageBox.RejectRole)

        choice_dialog.exec_()
        clicked_button = choice_dialog.clickedButton()

        if clicked_button == btn_cancel:
            return

        elif clicked_button == btn_file:
            # Upload single segmentation file
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Segmentation File", "", "NIfTI (*.nii *.nii.gz);;All Files (*)"
            )
            if path:
                self.seg_path = path
                self.lbl_seg_status.setText(
                    f"✅ Loaded: {os.path.basename(path)}")
                self.lbl_seg_status.setStyleSheet(
                    "color: #00ff00; font-weight: bold;")
                self.log_message(
                    f"Segmentation loaded: {os.path.basename(path)}")

                # Check if it's multi-label by loading it
                try:
                    nii = nib.load(path)
                    seg_data = nii.get_fdata()
                    unique_labels = np.unique(seg_data)
                    unique_labels = unique_labels[unique_labels > 0]

                    if len(unique_labels) > 1:
                        self.data_mode = 'segmentation_multilabel'
                        self.log_message(
                            f"🧠 Detected multi-label segmentation: {len(unique_labels)} structures")
                    else:
                        self.data_mode = 'segmentation'
                        self.log_message(
                            "Single structure segmentation detected")
                except Exception as e:
                    self.data_mode = 'segmentation'
                    self.log_message(
                        f"⚠️ Could not analyze file, assuming single structure: {e}")

        elif clicked_button == btn_folder:
            # Upload folder of segmentation files
            folder_path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select Folder with Segmentation Files (.nii)", ""
            )
            if folder_path:
                files = [f for f in os.listdir(folder_path)
                         if f.lower().endswith(('.nii', '.nii.gz'))]

                if not files:
                    QtWidgets.QMessageBox.warning(
                        self, "No Segmentation Files",
                        "No .nii or .nii.gz files found in this folder.")
                    return

                self.seg_path = folder_path  # Store folder path
                self.lbl_seg_status.setText(
                    f"✅ Loaded: {len(files)} files from folder")
                self.lbl_seg_status.setStyleSheet(
                    "color: #00ff00; font-weight: bold;")
                self.log_message(
                    f"Segmentation folder loaded: {len(files)} files")
                self.data_mode = 'segmentation_folder'

    def browse_model_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder with 3D Models", ""
        )
        if folder_path:
            files = [f for f in os.listdir(
                folder_path) if f.lower().endswith(('.obj', '.stl'))]
            if not files:
                QtWidgets.QMessageBox.warning(
                    self, "No Models", "No .obj or .stl files found.")
                return

            self.model_folder_path = folder_path
            self.lbl_model_status.setText(f"✅ Loaded: {len(files)} files")
            self.lbl_model_status.setStyleSheet(
                "color: #00ff00; font-weight: bold;")
            self.log_message(f"Models loaded: {len(files)} files")
            self.data_mode = 'obj_models'

    def on_show_anatomy_clicked(self):
        """Use anatomy transparency controller with brain multi-label support"""
        if not self.anatomy_controller:
            self.anatomy_controller = AnatomyTransparencyController(
                self.plotter, self.system_name, console_log=self.log_message
            )

        if self.data_mode == 'segmentation':
            self.anatomy_controller.load_from_segmentation(
                self.seg_path, build_heart_surfaces_from_seg
            )
        elif self.data_mode == 'segmentation_multilabel':
            # Handle multi-label brain segmentation (single file with multiple labels)
            self.load_brain_from_multilabel()
        elif self.data_mode == 'segmentation_folder':
            # Handle folder of separate segmentation files
            self.load_brain_from_folder()
        elif self.data_mode == 'obj_models':
            self.anatomy_controller.load_from_obj_folder(
                self.model_folder_path)
        else:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Upload data first!")
            return

        # Update current_surfaces reference
        self.current_surfaces = self.anatomy_controller.current_surfaces

        # Enable other features
        self.btn_focus.setEnabled(True)
        self.btn_flythrough.setEnabled(True)
        self.btn_clipping.setEnabled(True)
        if self.has_moving:
            self.btn_moving.setEnabled(True)
        self.btn_remove.setEnabled(True)

        # Auto-open transparency window
        QtCore.QTimer.singleShot(
            300, self.show_system_specific_transparency_window)

    def load_brain_from_folder(self):
        """Load brain structures from folder of separate .nii files"""
        self.log_message("\n" + "=" * 60)
        self.log_message("🧠 LOADING BRAIN FROM FOLDER OF SEGMENTATION FILES")
        self.log_message("=" * 60)

        try:
            files = [f for f in os.listdir(self.seg_path)
                     if f.lower().endswith(('.nii', '.nii.gz'))]

            self.log_message(f"📂 Found {len(files)} segmentation files")

            # Color palette
            colors = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#ABEBC6',
                '#FAD7A0', '#D7BDE2', '#A3E4D7', '#F9E79F', '#FADBD8',
                '#AED6F1', '#F8BBD0', '#FFCCBC', '#C5E1A5', '#FFECB3'
            ]

            self.current_surfaces = []

            # Process each file
            for idx, filename in enumerate(sorted(files)):
                filepath = os.path.join(self.seg_path, filename)
                self.log_message(f"\n🔄 Processing {filename}...")

                try:
                    # Load NIfTI file
                    nii = nib.load(filepath)
                    data = nii.get_fdata()

                    # Skip if empty
                    if data.max() == 0:
                        self.log_message(f"   ⚠️ Skipped (empty volume)")
                        continue

                    # Create binary mask
                    mask = (data > 0.5).astype(np.uint8)
                    voxel_count = np.sum(mask)

                    if voxel_count < 10:
                        self.log_message(
                            f"   ⚠️ Skipped (too small: {voxel_count} voxels)")
                        continue

                    self.log_message(f"   📊 Voxels: {voxel_count}")

                    # Create 3D mesh
                    self.log_message(f"   🔨 Creating mesh...")
                    verts, faces, _, _ = measure.marching_cubes(
                        mask, level=0.5)

                    # Convert to PyVista
                    faces_pv = np.hstack([[3] + list(f) for f in faces])
                    mesh = pv.PolyData(verts, faces_pv)
                    mesh = mesh.smooth(n_iter=50)

                    # Structure name from filename
                    structure_name = os.path.splitext(
                        os.path.splitext(filename)[0])[0]
                    structure_name = structure_name.replace('_', ' ').title()

                    # Assign color
                    color = colors[idx % len(colors)]

                    # Store surface
                    surface_dict = {
                        'name': structure_name,
                        'mesh': mesh,
                        'color': color,
                        'actor': None
                    }
                    self.current_surfaces.append(surface_dict)

                    self.log_message(
                        f"   ✅ {structure_name} created ({len(verts)} vertices)")

                except Exception as e:
                    self.log_message(f"   ❌ Failed: {e}")
                    continue

            if not self.current_surfaces:
                QtWidgets.QMessageBox.warning(
                    self, "Loading Failed",
                    "No valid brain structures found in folder.")
                return

            self.log_message("\n" + "=" * 60)
            self.log_message(
                f"✅ TOTAL STRUCTURES LOADED: {len(self.current_surfaces)}")
            self.log_message("=" * 60)

            # Update anatomy controller
            self.anatomy_controller.current_surfaces = self.current_surfaces
            self.anatomy_controller.render_surfaces()

        except Exception as e:
            self.log_message(f"\n❌ ERROR: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Loading Error",
                f"Failed to load brain structures:\n\n{str(e)}")
        """NEW: Load brain structures from multi-label segmentation file"""
        self.log_message("\n" + "=" * 60)
        self.log_message("🧠 LOADING BRAIN FROM MULTI-LABEL SEGMENTATION")
        self.log_message("=" * 60)

        try:
            # Load NIfTI file
            self.log_message("📖 Reading NIfTI file...")
            nii = nib.load(self.seg_path)
            seg_data = nii.get_fdata()

            self.log_message(f"✅ Volume shape: {seg_data.shape}")

            # Find unique labels
            unique_labels = np.unique(seg_data)
            unique_labels = unique_labels[unique_labels > 0]  # Skip background

            self.log_message(f"✅ Found {len(unique_labels)} brain structures")
            self.log_message(
                f"   Labels: {unique_labels[:10]}..." if len(unique_labels) > 10 else f"   Labels: {unique_labels}")

            # Color palette
            colors = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#ABEBC6',
                '#FAD7A0', '#D7BDE2', '#A3E4D7', '#F9E79F', '#FADBD8',
                '#AED6F1', '#F8BBD0', '#FFCCBC', '#C5E1A5', '#FFECB3'
            ]

            self.current_surfaces = []

            # Process each label
            for idx, label_value in enumerate(unique_labels):
                self.log_message(
                    f"\n🔄 Processing structure {int(label_value)}...")

                try:
                    # Extract this label's voxels
                    label_mask = (seg_data == label_value).astype(np.uint8)

                    # Check if enough voxels
                    voxel_count = np.sum(label_mask)
                    if voxel_count < 10:
                        self.log_message(
                            f"   ⚠️ Skipped (too small: {voxel_count} voxels)")
                        continue

                    self.log_message(f"   📊 Voxels: {voxel_count}")

                    # Create 3D mesh using marching cubes
                    self.log_message(f"   🔨 Creating 3D mesh...")
                    verts, faces, normals, values = measure.marching_cubes(
                        label_mask, level=0.5, step_size=1
                    )

                    # Convert to PyVista format
                    faces_pv = np.hstack([[3] + list(face) for face in faces])
                    mesh = pv.PolyData(verts, faces_pv)

                    # Smooth mesh
                    mesh = mesh.smooth(n_iter=50)

                    # Generate name
                    structure_name = f"Brain Structure {int(label_value)}"

                    # Assign color
                    color = colors[idx % len(colors)]

                    # Store surface
                    surface_dict = {
                        'name': structure_name,
                        'mesh': mesh,
                        'color': color,
                        'label': int(label_value),
                        'actor': None
                    }
                    self.current_surfaces.append(surface_dict)

                    self.log_message(
                        f"   ✅ {structure_name} created ({len(verts)} vertices)")

                except Exception as e:
                    self.log_message(f"   ❌ Failed: {e}")
                    continue

            if not self.current_surfaces:
                QtWidgets.QMessageBox.warning(
                    self, "No Structures",
                    "Could not extract any brain structures from segmentation.")
                return

            self.log_message("\n" + "=" * 60)
            self.log_message(
                f"✅ TOTAL STRUCTURES LOADED: {len(self.current_surfaces)}")
            self.log_message("=" * 60)

            # Update anatomy controller
            self.anatomy_controller.current_surfaces = self.current_surfaces
            self.anatomy_controller.render_surfaces()

        except Exception as e:
            self.log_message(f"\n❌ ERROR: {e}")
            QtWidgets.QMessageBox.critical(
                self, "Loading Error",
                f"Failed to load brain structures:\n\n{str(e)}")

    def on_selective_removal_clicked(self):
        """Launch selective removal dialog"""
        if not self.current_surfaces:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Show anatomy first!")
            return

        self.sync_surfaces_state()

        if not self.removal_controller:
            self.removal_controller = SelectiveRemovalController(
                self.plotter, self.system_name,
                self.current_surfaces, console_log=self.log_message
            )
        else:
            self.removal_controller.current_surfaces = self.current_surfaces

        self.removal_controller.show_removal_dialog(parent=self)

    def render_surfaces(self):
        """Use anatomy controller if available"""
        if self.anatomy_controller:
            self.anatomy_controller.current_surfaces = self.current_surfaces
            self.anatomy_controller.render_surfaces()
        else:
            self.plotter.clear()

            light1 = pv.Light(position=(1, 1, 1), light_type='scene light')
            light1.SetIntensity(1.5)
            self.plotter.add_light(light1)

            light2 = pv.Light(position=(-1, -1, -1), light_type='scene light')
            light2.SetIntensity(1.0)
            self.plotter.add_light(light2)

            for item in self.current_surfaces:
                stored_opacity = self.stored_opacities.get(item["name"], 0.98)
                actor = self.plotter.add_mesh(
                    item["mesh"], color=item["color"], opacity=stored_opacity,
                    smooth_shading=True, name=item["name"]
                )
                item['actor'] = actor

            self.plotter.add_axes()
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self.plotter.render()

            self.log_message("✅ Rendering complete!")

    def sync_surfaces_state(self):
        """Synchronize surfaces state from anatomy controller"""
        if self.anatomy_controller:
            self.current_surfaces = self.anatomy_controller.current_surfaces
            self.log_message(f"🔄 Synced {len(self.current_surfaces)} surfaces")
        else:
            self.log_message("⚠️ No anatomy controller to sync from")

    def show_system_specific_transparency_window(self):
        """Use anatomy controller"""
        if self.anatomy_controller:
            self.anatomy_controller.show_transparency_window(parent=self)

    def save_and_close_transparency_window(self):
        """Use anatomy controller"""
        if self.anatomy_controller:
            self.anatomy_controller.save_and_close_transparency_window()
            self.stored_opacities = self.anatomy_controller.stored_opacities

    def on_focus_navigation_clicked(self):
        if not self.current_surfaces:
            return
        self.focus_controller = FocusNavigationController(
            self.plotter, self.current_surfaces, console_log=self.log_message
        )
        self.show_focus_dialog()

    def show_focus_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"🔍 Focus - {self.system_name}")
        layout = QtWidgets.QVBoxLayout(dialog)

        combo = QtWidgets.QComboBox()
        combo.addItem("-- Select --")
        for surf in self.current_surfaces:
            combo.addItem(surf['name'])
        layout.addWidget(combo)

        btn_focus = QtWidgets.QPushButton("🔍 Focus")
        btn_focus.clicked.connect(
            lambda: self.apply_focus(combo.currentText(), dialog))
        layout.addWidget(btn_focus)

        btn_reset = QtWidgets.QPushButton("🔄 Reset")
        btn_reset.clicked.connect(lambda: self.reset_focus(dialog))
        layout.addWidget(btn_reset)

        dialog.exec_()

    def apply_focus(self, name, dialog):
        if name != "-- Select --":
            self.focus_controller.focus_on_structure(name)
            dialog.close()

    def reset_focus(self, dialog):
        if self.focus_controller:
            self.focus_controller.reset_focus()
        dialog.close()

    def on_clipping_clicked(self):
        """Show clipping mode selection"""
        if not self.current_surfaces:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "Show anatomy first!")
            return
        self.show_clipping_mode_dialog()

    def show_clipping_mode_dialog(self):
        """Dialog: 3D Object vs NIfTI Volume clipping"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("✂️ Clipping Mode")
        dialog.setMinimumWidth(500)

        layout = QtWidgets.QVBoxLayout(dialog)

        title = QtWidgets.QLabel("✂️ Choose Clipping Mode")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 18px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        obj_btn = QtWidgets.QPushButton(
            "🧊 3D Objects\n(Real-time mesh clipping)")
        obj_btn.setMinimumHeight(80)
        obj_btn.clicked.connect(lambda: self.launch_3d_clipping(dialog))
        layout.addWidget(obj_btn)

        nifti_btn = QtWidgets.QPushButton(
            "🧠 NIfTI Volume\n(Medical volume slicing)")
        nifti_btn.setMinimumHeight(80)
        nifti_btn.clicked.connect(lambda: self.launch_nifti_clipping(dialog))
        layout.addWidget(nifti_btn)

        cancel_btn = QtWidgets.QPushButton("✖ Cancel")
        cancel_btn.clicked.connect(dialog.close)
        layout.addWidget(cancel_btn)

        dialog.exec_()

    def launch_3d_clipping(self, dialog):
        """Launch 3D object clipping"""
        dialog.close()
        self.sync_surfaces_state()
        meshes = [surf['mesh'] for surf in self.current_surfaces]
        self.log_message(f"✂️ 3D clipping with {len(meshes)} meshes")

        if self.clipping_widget:
            try:
                self.clipping_widget.close()
            except:
                pass

        self.clipping_widget = ClippingControlWindow(self.plotter, meshes)
        self.clipping_widget.show()

    def launch_nifti_clipping(self, dialog):
        """Launch NIfTI volume clipping"""
        dialog.close()

        if self.volume_data is None:
            QtWidgets.QMessageBox.warning(
                self, "No Volume",
                "Upload a volume file first for NIfTI clipping."
            )
            return

        try:
            self.nifti_clipping_dialog = NIfTIClippingDialog(
                parent=self,
                plotter=self.plotter,
                volume_data=self.volume_data,
                current_surfaces=self.current_surfaces,
                log_callback=self.log_message
            )
            self.nifti_clipping_dialog.show()
            self.log_message("✅ NIfTI clipping opened!")

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"NIfTI clipping failed:\n{str(e)}")
            self.log_message(f"❌ Error: {e}")

    def on_curved_mpr_clicked(self):
        """Launch Curved MPR"""
        if self.volume_data is None:
            QtWidgets.QMessageBox.warning(
                self, "No Volume", "Upload volume first!")
            return

        self.log_message(f"\n🌊 Curved MPR - {self.system_name}")

        if not self.curved_mpr_controller:
            self.curved_mpr_controller = CurvedMPRController(
                self.plotter, self.volume_data, self.volume_affine, [],
                console_log=self.log_message
            )

        dialog = CurvedMPRDialog(self, self.curved_mpr_controller)
        dialog.show()

    def on_flythrough_clicked(self):
        """Show flythrough mode selection"""
        if not self.current_surfaces:
            QtWidgets.QMessageBox.warning(
                self, "No Surfaces", "Show anatomy first!")
            return

        self.flythrough_controller = FlythroughController(
            self.plotter, self.current_surfaces, console_log=self.log_message
        )

        if not self.manual_flythrough_controller:
            meshes = [surf['mesh'] for surf in self.current_surfaces]
            self.manual_flythrough_controller = ManualFlythroughController(
                self.plotter, meshes, console_log=self.log_message
            )

        self.show_flythrough_mode_dialog()

    """
    FIXED: Remove Custom Order Flythrough from Dental Tab
    Replace the show_flythrough_mode_dialog() method in GUI_Final.py or gui_last.py
    """

    def show_flythrough_mode_dialog(self):
        """Dialog: Automatic vs Manual vs Custom Order flythrough"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"🚀 Fly-through - {self.system_name}")
        dialog.setModal(True)
        dialog.setMinimumWidth(550)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        title = QtWidgets.QLabel("🚀 Choose Navigation Method")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 18px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        desc = QtWidgets.QLabel(
            "How do you want to navigate through the anatomy?")
        desc.setStyleSheet("color: #cccccc; font-size: 13px;")
        desc.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(desc)

        # Automatic path
        auto_btn = QtWidgets.QPushButton(
            "🎯 Automatic Path\n"
            "(Select structure → Auto-generate interior path)"
        )
        auto_btn.setMinimumHeight(80)
        auto_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #0f4c75, stop:1 #1b262c);
                color: white;
                border: 2px solid #00d4ff;
                border-left: 6px solid #00ff00;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #00d4ff, stop:1 #0f4c75);
            }
        """)
        auto_btn.clicked.connect(
            lambda: self.launch_automatic_flythrough(dialog))
        layout.addWidget(auto_btn)

        # Manual path
        manual_btn = QtWidgets.QPushButton(
            "✏️ Manual Path Drawing\n"
            "(CTRL+Click to draw custom path through anatomy)"
        )
        manual_btn.setMinimumHeight(80)
        manual_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #0f4c75, stop:1 #1b262c);
                color: white;
                border: 2px solid #00d4ff;
                border-left: 6px solid #ff00ff;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #00d4ff, stop:1 #0f4c75);
            }
        """)
        manual_btn.clicked.connect(
            lambda: self.launch_manual_flythrough(dialog))
        layout.addWidget(manual_btn)

        # FIXED: Custom order path - HIDE for Dental system
        if self.system_name != "Dental / Mouth":
            custom_btn = QtWidgets.QPushButton(
                "🎨 Custom Order Path\n"
                "(Reorder structures → Fly through in your sequence)"
            )
            custom_btn.setMinimumHeight(80)
            custom_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #0f4c75, stop:1 #1b262c);
                    color: white;
                    border: 2px solid #00d4ff;
                    border-left: 6px solid #ffaa00;
                    border-radius: 8px;
                    padding: 15px;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #00d4ff, stop:1 #0f4c75);
                }
            """)
            custom_btn.clicked.connect(
                lambda: self.launch_custom_order_flythrough(dialog))
            layout.addWidget(custom_btn)

        layout.addStretch()

        cancel_btn = QtWidgets.QPushButton("✖ Cancel")
        cancel_btn.setMinimumHeight(45)
        cancel_btn.clicked.connect(dialog.close)
        layout.addWidget(cancel_btn)

        dialog.exec_()

    def launch_custom_order_flythrough(self, dialog):
        """Launch custom order flythrough with structure reordering"""
        dialog.close()

        # Create controller if needed
        if not self.custom_order_controller:
            self.custom_order_controller = CustomOrderFlythroughController(
                self.plotter, self.current_surfaces, self.system_name,
                console_log=self.log_message
            )

        self.show_custom_order_dialog()

    def show_custom_order_dialog(self):
        """Dialog for custom structure ordering and animation"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(
            f"🎨 Custom Order Flythrough - {self.system_name}")
        dialog.setModal(False)
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(700)

        dialog.setStyleSheet("""
            QDialog {
                background-color: #1f2833;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #0f4c75, stop:1 #1b262c);
                color: white;
                border: 2px solid #00d4ff;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #00d4ff, stop:1 #0f4c75);
            }
            QListWidget {
                background-color: #0a0a0a;
                color: #00ff00;
                border: 2px solid #0f4c75;
                border-radius: 8px;
                padding: 5px;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #1f2833;
            }
            QListWidget::item:selected {
                background-color: #0f4c75;
                color: #00ffff;
            }
        """)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QtWidgets.QLabel(f"🎨 Custom Path Order - {self.system_name}")
        title.setStyleSheet(
            "color: #ffaa00; font-size: 16px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # Instructions
        instructions = QtWidgets.QLabel(
            "📋 Reorder structures to create your custom path:\n"
            "• Select a structure and use ↑↓ buttons to reorder\n"
            "• Default order follows anatomical flow\n"
            "• Camera will fly smoothly through structures in order"
        )
        instructions.setStyleSheet(
            "color: #cccccc; font-size: 11px; padding: 10px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Structure list
        list_label = QtWidgets.QLabel("Structure Order:")
        list_label.setStyleSheet(
            "color: #00d4ff; font-size: 12px; font-weight: bold;")
        layout.addWidget(list_label)

        self.custom_order_list = QtWidgets.QListWidget()
        self.custom_order_list.setMinimumHeight(300)

        # Populate with default order
        default_order = self.custom_order_controller.get_default_order()
        for i, struct in enumerate(default_order):
            self.custom_order_list.addItem(f"{i+1}. {struct}")

        layout.addWidget(self.custom_order_list)

        # Reorder buttons
        reorder_layout = QtWidgets.QHBoxLayout()

        btn_up = QtWidgets.QPushButton("↑ Move Up")
        btn_up.clicked.connect(lambda: self.move_structure_up())
        reorder_layout.addWidget(btn_up)

        btn_down = QtWidgets.QPushButton("↓ Move Down")
        btn_down.clicked.connect(lambda: self.move_structure_down())
        reorder_layout.addWidget(btn_down)

        btn_reset = QtWidgets.QPushButton("🔄 Reset to Default")
        btn_reset.clicked.connect(lambda: self.reset_structure_order())
        reorder_layout.addWidget(btn_reset)

        layout.addLayout(reorder_layout)

        # Speed control
        speed_group = QtWidgets.QGroupBox("⚡ Animation Speed")
        speed_layout = QtWidgets.QVBoxLayout()

        self.custom_speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.custom_speed_slider.setMinimum(20)
        self.custom_speed_slider.setMaximum(100)
        self.custom_speed_slider.setValue(50)
        speed_layout.addWidget(self.custom_speed_slider)

        self.custom_speed_label = QtWidgets.QLabel(
            "Speed: Medium (50ms/frame)")
        self.custom_speed_label.setStyleSheet(
            "color: #00d4ff; font-size: 11px;")
        self.custom_speed_label.setAlignment(QtCore.Qt.AlignCenter)
        speed_layout.addWidget(self.custom_speed_label)

        self.custom_speed_slider.valueChanged.connect(
            lambda v: self.custom_speed_label.setText(
                f"Speed: {'Fast' if v < 40 else 'Medium' if v < 70 else 'Slow'} ({v}ms/frame)")
        )

        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        # Action buttons
        btn_layout = QtWidgets.QHBoxLayout()

        btn_generate = QtWidgets.QPushButton("🎯 Generate Path")
        btn_generate.setMinimumHeight(50)
        btn_generate.clicked.connect(lambda: self.generate_custom_path())
        btn_layout.addWidget(btn_generate)

        btn_start = QtWidgets.QPushButton("▶️ Start Animation")
        btn_start.setMinimumHeight(50)
        btn_start.clicked.connect(lambda: self.start_custom_animation())
        btn_layout.addWidget(btn_start)

        btn_stop = QtWidgets.QPushButton("⏹️ Stop")
        btn_stop.setMinimumHeight(50)
        btn_stop.clicked.connect(lambda: self.stop_custom_animation())
        btn_layout.addWidget(btn_stop)

        layout.addLayout(btn_layout)

        # Close button
        btn_close = QtWidgets.QPushButton("✖ Close")
        btn_close.setMinimumHeight(45)
        btn_close.clicked.connect(dialog.close)
        layout.addWidget(btn_close)

        dialog.show()

    def move_structure_up(self):
        """Move selected structure up in the list"""
        current_row = self.custom_order_list.currentRow()
        if current_row > 0:
            item = self.custom_order_list.takeItem(current_row)
            self.custom_order_list.insertItem(current_row - 1, item)
            self.custom_order_list.setCurrentRow(current_row - 1)
            self._renumber_list()

    def move_structure_down(self):
        """Move selected structure down in the list"""
        current_row = self.custom_order_list.currentRow()
        if current_row < self.custom_order_list.count() - 1 and current_row >= 0:
            item = self.custom_order_list.takeItem(current_row)
            self.custom_order_list.insertItem(current_row + 1, item)
            self.custom_order_list.setCurrentRow(current_row + 1)
            self._renumber_list()

    def reset_structure_order(self):
        """Reset to default anatomical order"""
        self.custom_order_list.clear()
        default_order = self.custom_order_controller.get_default_order()
        for i, struct in enumerate(default_order):
            self.custom_order_list.addItem(f"{i+1}. {struct}")
        self.log_message("🔄 Reset to default anatomical order")

    def _renumber_list(self):
        """Renumber list items after reordering"""
        for i in range(self.custom_order_list.count()):
            item = self.custom_order_list.item(i)
            text = item.text()
            # Remove old number
            if '. ' in text:
                text = text.split('. ', 1)[1]
            # Add new number
            item.setText(f"{i+1}. {text}")

    def generate_custom_path(self):
        """Generate smooth path through custom ordered structures"""
        # Extract structure names from list
        ordered_structures = []
        for i in range(self.custom_order_list.count()):
            text = self.custom_order_list.item(i).text()
            # Remove numbering
            if '. ' in text:
                struct_name = text.split('. ', 1)[1]
            else:
                struct_name = text
            ordered_structures.append(struct_name)

        # Generate path
        success = self.custom_order_controller.generate_smooth_path(
            ordered_structures)

        if success:
            self.log_message("✅ Custom path generated successfully!")
        else:
            QtWidgets.QMessageBox.warning(
                self, "Path Generation Failed",
                "Could not generate path. Check console for details."
            )

    def start_custom_animation(self):
        """Start custom order flythrough animation"""
        speed = self.custom_speed_slider.value()
        success = self.custom_order_controller.start_animation(speed)

        if not success:
            QtWidgets.QMessageBox.warning(
                self, "Animation Failed",
                "Generate a path first before starting animation!"
            )

    def stop_custom_animation(self):
        """Stop custom order animation"""
        if self.custom_order_controller:
            self.custom_order_controller.stop_animation()

    def launch_automatic_flythrough(self, dialog):
        """Launch automatic flythrough"""
        dialog.close()
        self.show_flythrough_dialog()

    def launch_manual_flythrough(self, dialog):
        """Launch manual path drawing flythrough"""
        dialog.close()
        self.show_manual_flythrough_dialog()

    def show_manual_flythrough_dialog(self):
        """Dialog for manual path drawing and animation"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"✏️ Manual Fly-through - {self.system_name}")
        dialog.setModal(False)
        dialog.setMinimumWidth(550)
        dialog.setMinimumHeight(500)

        dialog.setStyleSheet("""
            QDialog {
                background-color: #1f2833;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #0f4c75, stop:1 #1b262c);
                color: white;
                border: 2px solid #00d4ff;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #00d4ff, stop:1 #0f4c75);
            }
            QGroupBox {
                background-color: #2a2a3e;
                border: 2px solid #ff00ff;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                font-weight: bold;
                color: #ff00ff;
            }
        """)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QtWidgets.QLabel(f"✏️ Manual Path Drawing - Virtual Endoscopy")
        title.setStyleSheet(
            "color: #ff00ff; font-size: 16px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        instructions_group = QtWidgets.QGroupBox("📋 How to Use")
        instructions_layout = QtWidgets.QVBoxLayout()

        instructions = QtWidgets.QLabel(
            "1️⃣ Click 'Start Drawing' to enter path drawing mode\n"
            "2️⃣ CTRL + Left Click on anatomy to place waypoints\n"
            "3️⃣ Right Click when finished to complete path\n"
            "4️⃣ Click 'Start Animation' to fly through your path\n\n"
            "💡 Draw a path through vessels, chambers, or any anatomy!\n"
            "💡 Camera will smoothly follow your drawn route"
        )
        instructions.setStyleSheet(
            "color: #e0e0e0; font-size: 11px; line-height: 1.6;")
        instructions.setWordWrap(True)
        instructions_layout.addWidget(instructions)
        instructions_group.setLayout(instructions_layout)
        layout.addWidget(instructions_group)

        status_group = QtWidgets.QGroupBox("📊 Path Status")
        status_layout = QtWidgets.QVBoxLayout()

        self.manual_status_label = QtWidgets.QLabel(
            "Ready. Click 'Start Drawing' to begin.")
        self.manual_status_label.setStyleSheet(
            "color: #00ff00; font-size: 12px; font-weight: bold;")
        self.manual_status_label.setAlignment(QtCore.Qt.AlignCenter)
        status_layout.addWidget(self.manual_status_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        speed_group = QtWidgets.QGroupBox("⚡ Animation Speed")
        speed_layout = QtWidgets.QVBoxLayout()

        self.manual_speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.manual_speed_slider.setMinimum(20)
        self.manual_speed_slider.setMaximum(100)
        self.manual_speed_slider.setValue(50)
        speed_layout.addWidget(self.manual_speed_slider)

        self.manual_speed_label = QtWidgets.QLabel(
            "Speed: Medium (50ms/frame)")
        self.manual_speed_label.setStyleSheet(
            "color: #00d4ff; font-size: 11px;")
        self.manual_speed_label.setAlignment(QtCore.Qt.AlignCenter)
        speed_layout.addWidget(self.manual_speed_label)

        self.manual_speed_slider.valueChanged.connect(
            lambda v: self.manual_speed_label.setText(
                f"Speed: {'Fast' if v < 40 else 'Medium' if v < 70 else 'Slow'} ({v}ms/frame)")
        )

        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        btn_layout1 = QtWidgets.QHBoxLayout()

        btn_start_draw = QtWidgets.QPushButton("✏️ Start Drawing")
        btn_start_draw.setMinimumHeight(50)
        btn_start_draw.clicked.connect(
            lambda: self.start_manual_drawing(dialog))
        btn_layout1.addWidget(btn_start_draw)

        btn_clear = QtWidgets.QPushButton("🗑️ Clear Path")
        btn_clear.setMinimumHeight(50)
        btn_clear.clicked.connect(lambda: self.clear_manual_path(dialog))
        btn_layout1.addWidget(btn_clear)

        layout.addLayout(btn_layout1)

        btn_layout2 = QtWidgets.QHBoxLayout()

        btn_animate = QtWidgets.QPushButton("▶️ Start Animation")
        btn_animate.setMinimumHeight(50)
        btn_animate.clicked.connect(
            lambda: self.start_manual_animation(dialog))
        btn_layout2.addWidget(btn_animate)

        btn_stop = QtWidgets.QPushButton("⏹️ Stop")
        btn_stop.setMinimumHeight(50)
        btn_stop.clicked.connect(lambda: self.stop_manual_animation(dialog))
        btn_layout2.addWidget(btn_stop)

        layout.addLayout(btn_layout2)

        btn_clear_path = QtWidgets.QPushButton("🗑️ Clear Path & Points")
        btn_clear_path.setMinimumHeight(50)
        btn_clear_path.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #ff8c00, stop:1 #cc7000);
                color: white;
                border: 2px solid #ffa500;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #ffa500, stop:1 #ff8c00);
                border: 2px solid #ffb732;
            }
        """)
        btn_clear_path.clicked.connect(lambda: self.clear_path_only(dialog))
        layout.addWidget(btn_clear_path)

        btn_reset = QtWidgets.QPushButton("🔄 Reset Everything")
        btn_reset.setMinimumHeight(50)
        btn_reset.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #8b0000, stop:1 #4a0000);
                color: white;
                border: 2px solid #ff4444;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #ff4444, stop:1 #8b0000);
                border: 2px solid #ff6666;
            }
        """)
        btn_reset.clicked.connect(
            lambda: self.reset_manual_flythrough_complete(dialog))
        layout.addWidget(btn_reset)

        btn_close = QtWidgets.QPushButton("✖ Close")
        btn_close.setMinimumHeight(45)
        btn_close.clicked.connect(dialog.close)
        layout.addWidget(btn_close)

        dialog.show()

    def start_manual_drawing(self, dialog):
        """Enable path drawing mode"""
        self.manual_flythrough_controller.enter_drawing_mode()
        self.manual_status_label.setText(
            "🎨 DRAWING MODE ACTIVE\nCTRL+Click to place waypoints | Right-click to finish")
        self.manual_status_label.setStyleSheet(
            "color: #ff00ff; font-size: 12px; font-weight: bold;")

    def clear_manual_path(self, dialog):
        """Clear the drawn path"""
        self.manual_flythrough_controller.stop_animation()
        self.manual_flythrough_controller.reset()
        self.manual_status_label.setText(
            "Path cleared. Ready to draw new path.")
        self.manual_status_label.setStyleSheet(
            "color: #00ff00; font-size: 12px; font-weight: bold;")

    def start_manual_animation(self, dialog):
        """Start flying through the manual path"""
        speed = self.manual_speed_slider.value()
        success = self.manual_flythrough_controller.start_animation(speed)

        if success:
            self.manual_status_label.setText(
                "🎬 ANIMATION RUNNING\nCamera flying through your path...")
            self.manual_status_label.setStyleSheet(
                "color: #00ffff; font-size: 12px; font-weight: bold;")
        else:
            self.manual_status_label.setText(
                "⚠️ No path to animate. Draw a path first!")
            self.manual_status_label.setStyleSheet(
                "color: #ff6600; font-size: 12px; font-weight: bold;")

    def stop_manual_animation(self, dialog):
        """Stop the animation"""
        self.manual_flythrough_controller.stop_animation()
        self.manual_status_label.setText(
            "⏹️ Animation stopped. Path preserved.")
        self.manual_status_label.setStyleSheet(
            "color: #00ff00; font-size: 12px; font-weight: bold;")

    def clear_path_only(self, dialog):
        """Just clear the drawn path and points"""
        try:
            if self.manual_flythrough_controller:
                self.manual_flythrough_controller.stop_animation()
                self.manual_flythrough_controller.reset()

            self.manual_status_label.setText(
                "🗑️ Path cleared. Ready to draw new path.")
            self.manual_status_label.setStyleSheet(
                "color: #ffa500; font-size: 12px; font-weight: bold;")
            self.log_message("🗑️ Path and points cleared!")

        except Exception as e:
            self.log_message(f"⚠️ Clear path error: {e}")
            QtWidgets.QMessageBox.warning(
                dialog, "Clear Warning", f"Error clearing path:\n{str(e)}")

    def reset_manual_flythrough_complete(self, dialog):
        """Complete reset: stop animation + clear path + reset camera"""
        reply = QtWidgets.QMessageBox.question(
            dialog, 'Reset Manual Flythrough',
            'This will:\n'
            '• Stop animation\n'
            '• Clear all drawn paths & points\n'
            '• Reset camera view\n\n'
            'Continue?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.No:
            return

        try:
            if self.manual_flythrough_controller:
                self.manual_flythrough_controller.stop_animation()
                self.manual_flythrough_controller.reset()

            self.plotter.reset_camera()
            self.plotter.render()

            self.manual_status_label.setText(
                "✅ Complete reset. Ready to draw new path.")
            self.manual_status_label.setStyleSheet(
                "color: #00ff00; font-size: 12px; font-weight: bold;")

            self.log_message("🔄 Manual flythrough completely reset!")
            QtWidgets.QMessageBox.information(
                dialog, "Reset Complete",
                "Manual flythrough has been completely reset.\n"
                "Camera view restored. You can draw a new path now."
            )

        except Exception as e:
            self.log_message(f"⚠️ Reset error: {e}")
            QtWidgets.QMessageBox.warning(
                dialog, "Reset Warning", f"Reset completed with issues:\n{str(e)}")

    def show_flythrough_dialog(self):
        """Automatic flythrough dialog"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"🚀 Automatic Fly-through - {self.system_name}")
        dialog.setModal(False)
        dialog.setMinimumWidth(600)

        layout = QtWidgets.QVBoxLayout(dialog)

        title = QtWidgets.QLabel(f"🚀 Navigate INSIDE {self.system_name}")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 16px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        select_layout = QtWidgets.QHBoxLayout()
        select_label = QtWidgets.QLabel("Select Structure:")
        select_layout.addWidget(select_label)

        self.flythrough_combo = QtWidgets.QComboBox()
        self.flythrough_combo.addItem("-- Select --")
        for surf in self.current_surfaces:
            self.flythrough_combo.addItem(surf['name'])
        select_layout.addWidget(self.flythrough_combo)
        layout.addLayout(select_layout)

        btn_generate = QtWidgets.QPushButton("🎯 Generate Path")
        btn_generate.setMinimumHeight(50)
        btn_generate.clicked.connect(self.generate_flythrough_path)
        layout.addWidget(btn_generate)

        btn_layout = QtWidgets.QHBoxLayout()

        btn_start = QtWidgets.QPushButton("▶️ Start")
        btn_start.clicked.connect(self.start_flythrough_animation)
        btn_layout.addWidget(btn_start)

        btn_stop = QtWidgets.QPushButton("⏹️ Stop")
        btn_stop.clicked.connect(self.stop_flythrough_animation)
        btn_layout.addWidget(btn_stop)

        layout.addLayout(btn_layout)

        btn_close = QtWidgets.QPushButton("✖ Close")
        btn_close.clicked.connect(dialog.close)
        layout.addWidget(btn_close)

        dialog.show()

    def generate_flythrough_path(self):
        name = self.flythrough_combo.currentText()
        if name != "-- Select --":
            self.flythrough_controller.generate_path_for_structure(name)
            self.log_message(f"✅ Path generated for {name}")

    def start_flythrough_animation(self):
        if self.flythrough_controller:
            self.flythrough_controller.start_animation(speed=50)

    def stop_flythrough_animation(self):
        if self.flythrough_controller:
            self.flythrough_controller.stop_animation()

    def on_moving_stuff_clicked(self):
        """Launch moving animation - heart or brain"""
        if not self.current_surfaces:
            return

        self.pump_controller = HeartPumpController(
            self.plotter, self.current_surfaces, console_log=self.log_message
        )

        if self.system_name == "Cardiovascular":
            if not self.pump_controller:
                self.pump_controller = HeartPumpController(
                    self.plotter, self.current_surfaces, console_log=self.log_message
                )
            self.show_pump_dialog()

        elif self.system_name == "Nervous":
            if not self.brain_controller:
                self.brain_controller = BrainAnimationController(
                    self.plotter, self.current_surfaces, console_log=self.log_message
                )
            self.show_brain_dialog()

    def show_pump_dialog(self):
        """Heart pumping animation dialog"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"💗 {self.system_name} Animation")
        layout = QtWidgets.QVBoxLayout(dialog)

        btn_start = QtWidgets.QPushButton("▶️ Start")
        btn_start.clicked.connect(
            lambda: self.pump_controller.start_animation())
        layout.addWidget(btn_start)

        btn_stop = QtWidgets.QPushButton("⏹️ Stop")
        btn_stop.clicked.connect(lambda: self.pump_controller.stop_animation())
        layout.addWidget(btn_stop)

        dialog.show()

    def show_brain_dialog(self):
        """Brain neural signal animation dialog"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"🧠 Neural Activity - {self.system_name}")
        dialog.setModal(False)
        dialog.setMinimumWidth(550)
        dialog.setMinimumHeight(500)

        dialog.setStyleSheet("""
            QDialog {
                background-color: #1f2833;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #0f4c75, stop:1 #1b262c);
                color: white;
                border: 2px solid #00d4ff;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #00d4ff, stop:1 #0f4c75);
            }
            QGroupBox {
                background-color: #2a2a3e;
                border: 2px solid #00d4ff;
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                font-weight: bold;
                color: #00d4ff;
            }
        """)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QtWidgets.QLabel("🧠 Neural Signal Propagation")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 18px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        desc_group = QtWidgets.QGroupBox("ℹ️ About Neural Activity")
        desc_layout = QtWidgets.QVBoxLayout()

        desc = QtWidgets.QLabel(
            "Watch electrical signals propagate through brain regions!\n\n"
            "The animation shows how neural activity travels along\n"
            "specific pathways during different cognitive tasks.\n\n"
            "• Cyan glow = Signal approaching\n"
            "• White flash = Signal peak\n"
            "• Returns to base color after signal passes"
        )
        desc.setStyleSheet(
            "color: #e0e0e0; font-size: 11px; line-height: 1.6;")
        desc.setWordWrap(True)
        desc_layout.addWidget(desc)
        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)

        pathway_group = QtWidgets.QGroupBox("🧭 Neural Pathways")
        pathway_layout = QtWidgets.QVBoxLayout()

        btn_thinking = QtWidgets.QPushButton(
            "💭 THINKING\n(Brainstem → Thalamus → Frontal → Motor)")
        btn_thinking.setMinimumHeight(70)
        btn_thinking.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #1e3a5f, stop:1 #0d1b2a);
                color: white;
                border: 2px solid #4a90e2;
                border-left: 6px solid #4a90e2;
                border-radius: 8px;
                padding: 15px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #4a90e2, stop:1 #1e3a5f);
                border: 2px solid #6bb6ff;
            }
        """)
        btn_thinking.clicked.connect(
            lambda: self.start_brain_animation("thinking", dialog))
        pathway_layout.addWidget(btn_thinking)

        btn_seeing = QtWidgets.QPushButton(
            "👁️ SEEING\n(Occipital → Temporal → Frontal)")
        btn_seeing.setMinimumHeight(70)
        btn_seeing.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #1e5f3a, stop:1 #0d2a1b);
                color: white;
                border: 2px solid #4ae290;
                border-left: 6px solid #4ae290;
                border-radius: 8px;
                padding: 15px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #4ae290, stop:1 #1e5f3a);
                border: 2px solid #6bffb6;
            }
        """)
        btn_seeing.clicked.connect(
            lambda: self.start_brain_animation("seeing", dialog))
        pathway_layout.addWidget(btn_seeing)

        btn_hearing = QtWidgets.QPushButton(
            "👂 HEARING\n(Temporal → Parietal → Frontal)")
        btn_hearing.setMinimumHeight(70)
        btn_hearing.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #5f1e5f, stop:1 #2a0d2a);
                color: white;
                border: 2px solid #e24ae2;
                border-left: 6px solid #e24ae2;
                border-radius: 8px;
                padding: 15px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #e24ae2, stop:1 #5f1e5f);
                border: 2px solid #ff6bff;
            }
        """)
        btn_hearing.clicked.connect(
            lambda: self.start_brain_animation("hearing", dialog))
        pathway_layout.addWidget(btn_hearing)

        pathway_group.setLayout(pathway_layout)
        layout.addWidget(pathway_group)

        control_group = QtWidgets.QGroupBox("🎮 Controls")
        control_layout = QtWidgets.QHBoxLayout()

        btn_stop = QtWidgets.QPushButton("⏹️ Stop Animation")
        btn_stop.setMinimumHeight(50)
        btn_stop.clicked.connect(lambda: self.stop_brain_animation(dialog))
        control_layout.addWidget(btn_stop)

        btn_close = QtWidgets.QPushButton("✖ Close")
        btn_close.setMinimumHeight(50)
        btn_close.clicked.connect(dialog.close)
        control_layout.addWidget(btn_close)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        self.brain_status_label = QtWidgets.QLabel(
            "Ready. Select a pathway to begin.")
        self.brain_status_label.setStyleSheet(
            "color: #00ff00; font-size: 12px; font-weight: bold; padding: 10px;")
        self.brain_status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.brain_status_label)

        dialog.show()

    def start_brain_animation(self, pathway, dialog):
        """Start neural signal animation"""
        if self.brain_controller:
            self.brain_controller.start_animation(pathway)

            pathway_names = {
                "thinking": "💭 THINKING",
                "seeing": "👁️ SEEING",
                "hearing": "👂 HEARING"
            }

            self.brain_status_label.setText(
                f"🧠 Active: {pathway_names.get(pathway, pathway.upper())}\n"
                f"Signal propagating through neural pathway..."
            )
            self.brain_status_label.setStyleSheet(
                "color: #00ffff; font-size: 12px; font-weight: bold; padding: 10px;")

    def stop_brain_animation(self, dialog):
        """Stop neural signal animation"""
        if self.brain_controller:
            self.brain_controller.stop_animation()

            self.brain_status_label.setText(
                "⏹️ Animation stopped. Ready for new pathway.")
            self.brain_status_label.setStyleSheet(
                "color: #ffaa00; font-size: 12px; font-weight: bold; padding: 10px;")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
