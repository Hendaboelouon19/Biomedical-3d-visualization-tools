"""
Anatomy Display and Transparency Control Module
Extracted from GUI.py for modular use

This module handles:
- Loading and rendering anatomical structures from segmentation or OBJ files
- System-specific transparency controls (Musculoskeletal, Nervous, Cardiovascular, Dental)
- Grouped transparency sliders with master controls
- Opacity persistence across features
"""

import os
import pyvista as pv
from PyQt5 import QtWidgets, QtCore


class AnatomyTransparencyController:
    """
    Controller for anatomy rendering and transparency management
    """

    def __init__(self, plotter, system_name, console_log=None):
        """
        Initialize the controller

        Args:
            plotter: PyVista QtInteractor plotter instance
            system_name: Name of the anatomical system (e.g., "Cardiovascular")
            console_log: Callback function for logging messages
        """
        self.plotter = plotter
        self.system_name = system_name
        self.console_log = console_log or print

        # Data storage
        self.current_surfaces = []
        self.stored_opacities = {}  # Persistent opacity settings

        # UI references
        self.transparency_window = None
        self.transparency_sliders = {}

    def log_message(self, msg):
        """Log a message using the provided callback"""
        self.console_log(msg)

    # ==================== ANATOMY LOADING ====================

    def load_from_segmentation(self, seg_path, build_surfaces_func):
        """
        Load anatomy from segmentation file

        Args:
            seg_path: Path to segmentation .nii/.nii.gz file
            build_surfaces_func: Function to build surfaces from segmentation
        """
        self.log_message(
            f"\n🫀 Building {self.system_name} from segmentation...")
        surfaces = build_surfaces_func(seg_path, console_log=self.console_log)
        self.current_surfaces = surfaces
        self.render_surfaces()

    def load_from_obj_folder(self, folder_path):
        """
        Load anatomy from folder containing OBJ/STL files

        Args:
            folder_path: Path to folder with .obj/.stl files
        """
        self.log_message(f"\n🫀 Loading {self.system_name} from OBJ files...")
        surfaces = self._load_obj_models_from_folder(folder_path)
        self.current_surfaces = surfaces
        self.render_surfaces()

    def _load_obj_models_from_folder(self, folder_path):
        """Load all OBJ/STL files from folder with color assignment"""
        surfaces = []

        # Color mapping based on filename keywords
        color_map = {
            'aorta': '#FF5050', 'pulmonary': '#FF69B4',
            'left_ventricle': '#FF0000', 'lv': '#FF0000',
            'right_ventricle': '#8B0000', 'rv': '#8B0000',
            'left_atrium': '#FFB6C1', 'la': '#FFB6C1',
            'right_atrium': '#DC143C', 'ra': '#DC143C',
            'myocardium': '#CD5C5C',
        }

        files = sorted([f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.obj', '.stl'))])

        for filename in files:
            filepath = os.path.join(folder_path, filename)
            try:
                mesh = pv.read(filepath)
                name = os.path.splitext(filename)[0].replace('_', ' ').title()

                # Assign color based on filename
                color = '#888888'
                for keyword, assigned_color in color_map.items():
                    if keyword in filename.lower():
                        color = assigned_color
                        break

                surfaces.append({'name': name, 'mesh': mesh, 'color': color})
            except Exception as e:
                self.log_message(f"⚠️ Could not load {filename}: {e}")

        return surfaces

    def render_surfaces(self):
        """Render all loaded surfaces in the plotter"""
        self.plotter.clear()

        # Add lighting for better visualization
        light1 = pv.Light(position=(1, 1, 1), light_type='scene light')
        light1.SetIntensity(1.5)
        self.plotter.add_light(light1)

        light2 = pv.Light(position=(-1, -1, -1), light_type='scene light')
        light2.SetIntensity(1.0)
        self.plotter.add_light(light2)

        # Render each surface
        for item in self.current_surfaces:
            # Check if we have stored opacity for this structure
            stored_opacity = self.stored_opacities.get(item["name"], 0.95)

            # Add mesh and store actor reference
            actor = self.plotter.add_mesh(
                item["mesh"],
                color=item["color"],
                opacity=stored_opacity,
                smooth_shading=True,
                name=item["name"]
            )
            # CRITICAL: Store the actor reference
            item['actor'] = actor

        self.plotter.add_axes()
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

        self.log_message("✅ Rendering complete!")

    # ==================== STRUCTURE CATEGORIZATION ====================

    def categorize_musculoskeletal_structures(self):
        """Categorize structures for musculoskeletal system"""
        categories = {
            'bones': {
                'femur': [], 'tibia': [], 'fibula': [], 'patella': [],
                'talus': [], 'calcaneus': [], 'foot_bones': [], 'other_bones': []
            },
            'muscles': {
                'soleus': [], 'tibialis': [], 'semitendinosus': [], 'other_muscles': []
            }
        }

        bone_keywords = {
            'femur': ['femur'], 'tibia': ['tibia'], 'fibula': ['fibula'],
            'patella': ['patella'], 'talus': ['talus'], 'calcaneus': ['calcaneus'],
            'foot_bones': ['metatarsal', 'phalanx', 'cuneiform', 'cuboid', 'navicular', 'sesamoid']
        }

        muscle_keywords = {
            'soleus': ['soleus'], 'tibialis': ['tibialis'], 'semitendinosus': ['semitendinosus']
        }

        for surf in self.current_surfaces:
            name_lower = surf['name'].lower()
            categorized = False

            # Check bones
            for bone_type, keywords in bone_keywords.items():
                if any(keyword in name_lower for keyword in keywords):
                    categories['bones'][bone_type].append(surf)
                    categorized = True
                    break

            # Check muscles
            if not categorized:
                for muscle_type, keywords in muscle_keywords.items():
                    if any(keyword in name_lower for keyword in keywords):
                        categories['muscles'][muscle_type].append(surf)
                        categorized = True
                        break

            # Default categorization
            if not categorized:
                if any(bone_word in name_lower for bone_word in ['bone', 'phalanx', 'metatarsal']):
                    categories['bones']['other_bones'].append(surf)
                else:
                    categories['muscles']['other_muscles'].append(surf)

        return categories

    def categorize_nervous_structures(self):
        """Categorize structures for nervous system"""
        categories = {
            'skull_bones': {
                'frontal': [], 'parietal': [], 'temporal': [], 'occipital': [],
                'sphenoid': [], 'ethmoid': [], 'zygomatic': [], 'maxilla': [],
                'palatine': [], 'other_skull': []
            },
            'brain_structures': []
        }

        skull_keywords = {
            'frontal': ['frontal'], 'parietal': ['parietal'], 'temporal': ['temporal'],
            'occipital': ['occipital'], 'sphenoid': ['sphenoid'], 'ethmoid': ['ethmoid'],
            'zygomatic': ['zygomatic'], 'maxilla': ['maxilla'], 'palatine': ['palatine']
        }

        brain_keywords = ['gyrus', 'nucleus', 'ventricle', 'amygdala', 'hippocampus',
                          'cerebellum', 'commissure', 'fornix', 'stria', 'capsule']

        for surf in self.current_surfaces:
            name_lower = surf['name'].lower()

            # Check if it's a brain structure
            if any(keyword in name_lower for keyword in brain_keywords):
                categories['brain_structures'].append(surf)
                continue

            # Check skull bones
            categorized = False
            for bone_type, keywords in skull_keywords.items():
                if any(keyword in name_lower for keyword in keywords):
                    categories['skull_bones'][bone_type].append(surf)
                    categorized = True
                    break

            if not categorized:
                if 'bone' in name_lower or 'atlas' in name_lower or 'axis' in name_lower:
                    categories['skull_bones']['other_skull'].append(surf)
                else:
                    categories['brain_structures'].append(surf)

        return categories

    def categorize_cardiovascular_structures(self):
        """Categorize structures for cardiovascular system"""
        categories = {'ribs': [], 'heart_structures': []}

        rib_keywords = ['rib', 'first rib', 'second rib', 'third rib']

        for surf in self.current_surfaces:
            name_lower = surf['name'].lower()
            if any(keyword in name_lower for keyword in rib_keywords):
                categories['ribs'].append(surf)
            else:
                categories['heart_structures'].append(surf)

        return categories

    def categorize_dental_structures(self):
        """Categorize structures for dental system"""
        categories = {'jaw': [], 'teeth': []}

        jaw_keywords = ['mandible', 'maxilla', 'jaw', 'palatine']
        teeth_keywords = ['tooth', 'teeth',
                          'incisor', 'canine', 'molar', 'premolar']

        for surf in self.current_surfaces:
            name_lower = surf['name'].lower()
            if any(keyword in name_lower for keyword in jaw_keywords):
                categories['jaw'].append(surf)
            elif any(keyword in name_lower for keyword in teeth_keywords):
                categories['teeth'].append(surf)
            else:
                categories['jaw'].append(surf)

        return categories

    # ==================== TRANSPARENCY WINDOW ====================

    def show_transparency_window(self, parent=None):
        """Show appropriate transparency window based on system type"""
        if not self.current_surfaces:
            return

        if self.system_name == "Musculoskeletal":
            self._show_musculoskeletal_transparency(parent)
        elif self.system_name == "Nervous":
            self._show_nervous_transparency(parent)
        elif self.system_name == "Cardiovascular":
            self._show_cardiovascular_transparency(parent)
        elif self.system_name == "Dental / Mouth":
            self._show_dental_transparency(parent)

    def _show_musculoskeletal_transparency(self, parent):
        """Create transparency window for musculoskeletal system"""
        categories = self.categorize_musculoskeletal_structures()

        self.transparency_window = QtWidgets.QDialog(parent)
        self.transparency_window.setWindowTitle(
            "💪 Musculoskeletal Transparency Controls")
        self.transparency_window.setModal(False)
        self.transparency_window.setMinimumWidth(500)
        self.transparency_window.setMinimumHeight(700)

        self._apply_transparency_window_style()

        main_layout = QtWidgets.QVBoxLayout(self.transparency_window)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QtWidgets.QLabel("🦴 Musculoskeletal Transparency Controls")
        title.setStyleSheet(
            "color: #ecf0f1; font-size: 20px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        # Create tabs
        tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(tab_widget)

        self.transparency_sliders = {}

        # BONES TAB
        bones_tab = self._create_bones_tab(categories)
        tab_widget.addTab(bones_tab, "🦴 Bones")

        # MUSCLES TAB
        muscles_tab = self._create_muscles_tab(categories)
        tab_widget.addTab(muscles_tab, "💪 Muscles")

        # Done button
        self._add_done_button(main_layout)

        self.transparency_window.show()
        self.log_message("🦴 Transparency controls opened")

    def _create_bones_tab(self, categories):
        """Create bones tab for musculoskeletal system"""
        bones_tab = QtWidgets.QWidget()
        bones_layout = QtWidgets.QVBoxLayout(bones_tab)
        bones_layout.setSpacing(5)

        # Master bone control
        master_bone_group = self._create_master_control_group(
            "ALL BONES", categories['bones'], '#e74c3c')
        bones_layout.addWidget(master_bone_group)

        # Individual bone groups
        bone_groups = {
            'Femur': categories['bones']['femur'],
            'Tibia': categories['bones']['tibia'],
            'Fibula': categories['bones']['fibula'],
            'Patella': categories['bones']['patella'],
            'Talus': categories['bones']['talus'],
            'Calcaneus': categories['bones']['calcaneus'],
            'Foot Bones': categories['bones']['foot_bones'],
            'Other Bones': categories['bones']['other_bones']
        }

        for group_name, structures in bone_groups.items():
            if structures:
                group_widget = self._create_group_control(
                    group_name, structures)
                bones_layout.addWidget(group_widget)

        bones_layout.addStretch()
        return bones_tab

    def _create_muscles_tab(self, categories):
        """Create muscles tab for musculoskeletal system"""
        muscles_tab = QtWidgets.QWidget()
        muscles_layout = QtWidgets.QVBoxLayout(muscles_tab)
        muscles_layout.setSpacing(5)

        # Master muscle control
        master_muscle_group = self._create_master_control_group(
            "ALL MUSCLES", categories['muscles'], '#e67e22')
        muscles_layout.addWidget(master_muscle_group)

        # Individual muscle groups
        muscle_groups = {
            'Soleus': categories['muscles']['soleus'],
            'Tibialis': categories['muscles']['tibialis'],
            'Semitendinosus': categories['muscles']['semitendinosus'],
            'Other Muscles': categories['muscles']['other_muscles']
        }

        for group_name, structures in muscle_groups.items():
            if structures:
                group_widget = self._create_group_control(
                    group_name, structures)
                muscles_layout.addWidget(group_widget)

        muscles_layout.addStretch()
        return muscles_tab

    def _show_nervous_transparency(self, parent):
        """Create transparency window for nervous system"""
        categories = self.categorize_nervous_structures()

        self.transparency_window = QtWidgets.QDialog(parent)
        self.transparency_window.setWindowTitle(
            "🧠 Nervous System Transparency Controls")
        self.transparency_window.setModal(False)
        self.transparency_window.setMinimumWidth(500)
        self.transparency_window.setMinimumHeight(700)

        self._apply_transparency_window_style('#9b59b6')

        main_layout = QtWidgets.QVBoxLayout(self.transparency_window)
        main_layout.setContentsMargins(15, 15, 15, 15)

        title = QtWidgets.QLabel(
            "🧠 Nervous System - Skull Transparency Controls")
        title.setStyleSheet(
            "color: #ecf0f1; font-size: 20px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)

        self.transparency_sliders = {}

        # All skull bones master control
        all_skull_bones = []
        for bone_list in categories['skull_bones'].values():
            all_skull_bones.extend(bone_list)

        if all_skull_bones:
            master_group = self._create_master_control_group(
                "ALL SKULL BONES", {'all': all_skull_bones}, '#e74c3c')
            scroll_layout.addWidget(master_group)

            # Individual bones
            bone_groups = {
                '💀 Frontal Bone': categories['skull_bones']['frontal'],
                '💀 Parietal Bones': categories['skull_bones']['parietal'],
                '💀 Temporal Bones': categories['skull_bones']['temporal'],
                '💀 Occipital Bone': categories['skull_bones']['occipital'],
                '💀 Sphenoid Bone': categories['skull_bones']['sphenoid'],
                '💀 Ethmoid Bone': categories['skull_bones']['ethmoid'],
                '💀 Zygomatic Bones': categories['skull_bones']['zygomatic'],
                '💀 Maxilla': categories['skull_bones']['maxilla'],
                '💀 Palatine Bones': categories['skull_bones']['palatine'],
                '💀 Other Skull Bones': categories['skull_bones']['other_skull']
            }

            for group_name, structures in bone_groups.items():
                if structures:
                    group_widget = self._create_group_control(
                        group_name, structures)
                    scroll_layout.addWidget(group_widget)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self._add_done_button(main_layout)
        self.transparency_window.show()
        self.log_message("💀 Skull transparency controls opened")

    def _show_cardiovascular_transparency(self, parent):
        """Create transparency window for cardiovascular system"""
        categories = self.categorize_cardiovascular_structures()

        self.transparency_window = QtWidgets.QDialog(parent)
        self.transparency_window.setWindowTitle(
            "❤️ Cardiovascular Transparency Controls")
        self.transparency_window.setModal(False)
        self.transparency_window.setMinimumWidth(500)
        self.transparency_window.setMinimumHeight(400)

        self._apply_transparency_window_style('#e74c3c')

        main_layout = QtWidgets.QVBoxLayout(self.transparency_window)
        main_layout.setContentsMargins(15, 15, 15, 15)

        title = QtWidgets.QLabel(
            "❤️ Cardiovascular - Rib Transparency Controls")
        title.setStyleSheet(
            "color: #ecf0f1; font-size: 20px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        self.transparency_sliders = {}

        if categories['ribs']:
            rib_group = self._create_master_control_group(
                "ALL RIBS", {'ribs': categories['ribs']}, '#e74c3c')
            main_layout.addWidget(rib_group)

            info_label = QtWidgets.QLabel(
                f"ℹ️ Controlling {len(categories['ribs'])} rib structures together\n"
                f"💡 Heart structures remain at full opacity"
            )
            info_label.setStyleSheet(
                "color: #ecf0f1; font-size: 11px; font-style: italic;")
            info_label.setAlignment(QtCore.Qt.AlignCenter)
            main_layout.addWidget(info_label)

        main_layout.addStretch()
        self._add_done_button(main_layout)
        self.transparency_window.show()
        self.log_message("🦴 Rib transparency controls opened")

    def _show_dental_transparency(self, parent):
        """Create transparency window for dental system"""
        categories = self.categorize_dental_structures()

        self.transparency_window = QtWidgets.QDialog(parent)
        self.transparency_window.setWindowTitle(
            "🦷 Dental Transparency Controls")
        self.transparency_window.setModal(False)
        self.transparency_window.setMinimumWidth(500)
        self.transparency_window.setMinimumHeight(400)

        self._apply_transparency_window_style('#f39c12')

        main_layout = QtWidgets.QVBoxLayout(self.transparency_window)
        main_layout.setContentsMargins(15, 15, 15, 15)

        title = QtWidgets.QLabel("🦷 Dental System - Jaw Transparency Controls")
        title.setStyleSheet(
            "color: #ecf0f1; font-size: 20px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        main_layout.addWidget(title)

        self.transparency_sliders = {}

        if categories['jaw']:
            jaw_group = self._create_master_control_group(
                "JAW TRANSPARENCY", {'jaw': categories['jaw']}, '#f39c12')
            main_layout.addWidget(jaw_group)

        main_layout.addStretch()
        self._add_done_button(main_layout)
        self.transparency_window.show()
        self.log_message("🦷 Jaw transparency controls opened")

    # ==================== UI HELPER METHODS ====================

    def _apply_transparency_window_style(self, accent_color='#3498db'):
        """Apply styling to transparency window"""
        self.transparency_window.setStyleSheet(f"""
            QDialog {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #2c3e50, stop:1 #34495e);
            }}
            QGroupBox {{
                background-color: #ecf0f1;
                border: 3px solid {accent_color};
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }}
            QLabel {{
                color: #2c3e50;
                font-size: 12px;
                font-weight: 600;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid #bdc3c7;
                height: 8px;
                background: #ecf0f1;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {accent_color};
                border: 2px solid {accent_color};
                width: 22px;
                height: 22px;
                margin: -7px 0;
                border-radius: 11px;
            }}
            QPushButton {{
                background-color: {accent_color};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: bold;
            }}
            QTabWidget::pane {{
                border: 2px solid {accent_color};
                border-radius: 8px;
                background-color: #ecf0f1;
            }}
            QTabBar::tab {{
                background-color: #95a5a6;
                color: white;
                padding: 12px 24px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background-color: {accent_color};
            }}
        """)

    def _create_master_control_group(self, title, category_dict, color):
        """Create master control group for multiple structures"""
        group = QtWidgets.QGroupBox(f"🎚️ {title}")
        layout = QtWidgets.QVBoxLayout()

        all_structures = []
        for structures_list in category_dict.values():
            all_structures.extend(structures_list)

        if not all_structures:
            label = QtWidgets.QLabel("No structures in this category")
            label.setStyleSheet("color: #95a5a6; font-style: italic;")
            layout.addWidget(label)
            group.setLayout(layout)
            return group

        # Header
        header = QtWidgets.QHBoxLayout()
        count_label = QtWidgets.QLabel(
            f"Control {len(all_structures)} structures together")
        count_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        header.addWidget(count_label)
        header.addStretch()

        value_label = QtWidgets.QLabel("95%")
        value_label.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 14px;")
        header.addWidget(value_label)
        layout.addLayout(header)

        # Slider
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(95)

        slider.valueChanged.connect(
            lambda value, structures=all_structures, lbl=value_label:
            self._update_group_transparency(structures, value, lbl)
        )

        layout.addWidget(slider)

        self.transparency_sliders[f"MASTER_{title}"] = {
            'slider': slider,
            'label': value_label,
            'structures': all_structures
        }

        group.setLayout(layout)
        return group

    def _create_group_control(self, group_name, structures):
        """Create control for specific structure group"""
        group = QtWidgets.QGroupBox(group_name)
        layout = QtWidgets.QVBoxLayout()

        if not structures:
            return group

        # Header
        header = QtWidgets.QHBoxLayout()
        count_label = QtWidgets.QLabel(f"{len(structures)} part(s)")
        count_label.setStyleSheet("color: #95a5a6; font-size: 10px;")
        header.addWidget(count_label)
        header.addStretch()

        value_label = QtWidgets.QLabel("95%")
        value_label.setStyleSheet("color: #16a085; font-weight: bold;")
        header.addWidget(value_label)
        layout.addLayout(header)

        # Slider
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(95)

        slider.valueChanged.connect(
            lambda value, structs=structures, lbl=value_label:
            self._update_group_transparency(structs, value, lbl)
        )

        layout.addWidget(slider)

        self.transparency_sliders[group_name] = {
            'slider': slider,
            'label': value_label,
            'structures': structures
        }

        group.setLayout(layout)
        return group

    def _update_group_transparency(self, structures, opacity_percent, label):
        """Update opacity for group of structures"""
        opacity = opacity_percent / 100.0
        label.setText(f"{opacity_percent}%")

        updated_count = 0
        for surf in structures:
            try:
                if 'actor' in surf and surf['actor'] is not None:
                    surf['actor'].GetProperty().SetOpacity(opacity)
                    updated_count += 1
            except Exception as e:
                self.log_message(f"⚠️ Could not update {surf['name']}: {e}")

        if updated_count > 0:
            self.plotter.render()
            if updated_count == 1:
                self.log_message(
                    f"✅ Updated {structures[0]['name']}: {opacity_percent}%")
            else:
                self.log_message(
                    f"✅ Updated {updated_count} structures: {opacity_percent}%")

    def _add_done_button(self, layout):
        """Add Done button to transparency window"""
        btn_done = QtWidgets.QPushButton("✅ Done (Save Settings)")
        btn_done.setMinimumHeight(50)
        btn_done.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        btn_done.clicked.connect(self.save_and_close_transparency_window)
        layout.addWidget(btn_done)

    def save_and_close_transparency_window(self):
        """Save opacity settings and close window"""
        for surf in self.current_surfaces:
            if 'actor' in surf and surf['actor'] is not None:
                try:
                    current_opacity = surf['actor'].GetProperty().GetOpacity()
                    self.stored_opacities[surf['name']] = current_opacity
                except Exception as e:
                    self.log_message(
                        f"⚠️ Could not save opacity for {surf['name']}: {e}")

        saved_count = len(self.stored_opacities)
        self.log_message(f"✅ Saved {saved_count} opacity settings to memory")
        self.log_message("💾 Settings will persist across all features")

        if self.transparency_window:
            self.transparency_window.close()
            self.transparency_window = None


# ==================== INTEGRATION INSTRUCTIONS ====================

"""
HOW TO INTEGRATE INTO GUI.py:

1. IMPORT THE MODULE at the top of GUI.py:
   
   from anatomy_transparency_module import AnatomyTransparencyController

2. IN SystemTab.__init__(), ADD THIS CONTROLLER:
   
   # Add after line: self.data_mode = None
   self.anatomy_controller = None

3. REPLACE on_show_anatomy_clicked() METHOD:
   
   def on_show_anatomy_clicked(self):
       if not self.anatomy_controller:
           self.anatomy_controller = AnatomyTransparencyController(
               self.plotter, self.system_name, console_log=self.log_message
           )
       
       if self.data_mode == 'segmentation':
           from feature_show_anatomy import build_heart_surfaces_from_seg
           self.anatomy_controller.load_from_segmentation(
               self.seg_path, build_heart_surfaces_from_seg
           )
       elif self.data_mode == 'obj_models':
           self.anatomy_controller.load_from_obj_folder(self.model_folder_path)
       else:
           QtWidgets.QMessageBox.warning(self, "No Data", "Upload data first!")
           return
       
       # Update current_surfaces reference
       self.current_surfaces = self.anatomy_controller.current_surfaces
       
       # Enable other features
       self.btn_focus.setEnabled(True)
       self.btn_flythrough.setEnabled(True)
       self.btn_clipping.setEnabled(True)
       if self.has_moving:
           self.btn_moving.setEnabled(True)
       
       # Auto-open transparency window
       QtCore.QTimer.singleShot(300, self.show_system_specific_transparency_window)

4. REPLACE show_system_specific_transparency_window() METHOD:
   
   def show_system_specific_transparency_window(self):
       if self.anatomy_controller:
           self.anatomy_controller.show_transparency_window(parent=self)

5. REPLACE save_and_close_transparency_window() METHOD:
   
   def save_and_close_transparency_window(self):
       if self.anatomy_controller:
           self.anatomy_controller.save_and_close_transparency_window()
           # Update stored opacities reference
           self.stored_opacities = self.anatomy_controller.stored_opacities

6. UPDATE render_surfaces() METHOD:
   
   def render_surfaces(self):
       # If anatomy controller exists, use it
       if self.anatomy_controller:
           self.anatomy_controller.current_surfaces = self.current_surfaces
           self.anatomy_controller.render_surfaces()
       else:
           # Original render code as fallback
           self.plotter.clear()
           # ... rest of original code

7. REMOVE THESE OLD METHODS FROM SystemTab (they're now in the module):
   - show_anatomy_from_segmentation()
   - show_anatomy_from_obj_models()
   - load_obj_models_from_folder()
   - categorize_musculoskeletal_structures()
   - categorize_nervous_structures()
   - categorize_cardiovascular_structures()
   - categorize_dental_structures()
   - show_musculoskeletal_transparency_window()
   - show_nervous_transparency_window()
   - show_cardiovascular_transparency_window()
   - show_dental_transparency_window()
   - create_master_control_group()
   - create_group_control()
   - update_group_transparency()
   - _create_bones_tab()
   - _create_muscles_tab()

8. BENEFITS OF THIS MODULAR APPROACH:
   ✅ Cleaner code organization
   ✅ Easier to maintain and debug
   ✅ Reusable in other projects
   ✅ Separated concerns (UI vs. Logic)
   ✅ Easier to test independently
   ✅ Reduced GUI.py file size (~800 lines removed)

9. FILE STRUCTURE:
   project/
   ├── GUI.py (main application)
   ├── anatomy_transparency_module.py (NEW - this file)
   ├── feature_show_anatomy.py
   ├── feature_focus_navigation.py
   └── ... (other feature files)

10. TESTING THE INTEGRATION:
    - Run GUI.py normally
    - Upload segmentation or OBJ files
    - Click "Show Anatomy" button
    - Transparency window should open automatically
    - All sliders should work as before
    - Opacity settings should persist across features

11. ADDITIONAL NOTES:
    - The module is fully self-contained
    - No dependencies on GUI.py internals
    - Uses dependency injection (plotter, system_name)
    - All callbacks handled through log_message parameter
    - Stored opacities persist in the controller instance

12. TROUBLESHOOTING:
    If transparency doesn't work:
    - Check that anatomy_controller is initialized
    - Verify current_surfaces is synced between GUI and controller
    - Ensure actor references are stored in surface dictionaries
    - Check that plotter.render() is called after opacity changes

END OF INTEGRATION INSTRUCTIONS
"""
