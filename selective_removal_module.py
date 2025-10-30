"""
Selective Structure Removal Module
Allows PERMANENT removal of specific anatomical groups per system

Features:
- System-specific removal rules (ribs, skull, leg bones, gums)
- CODE-BASED matching for precise identification (e.g., MM=skull, FJ=brain)
- PERMANENTLY deletes from current_surfaces list (won't appear in other functions)
- Restore removed structures if needed
- Visual feedback in console

Code Matching Examples:
- Nervous System: MM codes = skull bones, FJ codes = brain structures
- Cardiovascular: Rib codes in filenames
- Other systems: Can use keyword matching as fallback
"""

from PyQt5 import QtWidgets, QtCore


class SelectiveRemovalController:
    """
    Controller for selectively hiding/removing anatomical structures
    """

    def __init__(self, plotter, system_name, current_surfaces_ref, console_log=None):
        """
        Initialize the controller

        Args:
            plotter: PyVista QtInteractor plotter instance
            system_name: Name of the anatomical system
            current_surfaces_ref: REFERENCE to the current_surfaces list (will modify in place)
            console_log: Callback function for logging messages
        """
        self.plotter = plotter
        self.system_name = system_name
        self.current_surfaces_ref = current_surfaces_ref  # Reference to GUI's list
        self.console_log = console_log or print

        # Track removed structures (for restore functionality)
        self.removed_structures = []
        self.removal_dialog = None

    def log_message(self, msg):
        """Log a message using the provided callback"""
        self.console_log(msg)

    # ==================== CATEGORIZATION RULES ====================

    def get_removal_groups(self):
        """
        Get removal groups based on system type
        Returns: dict with group names and their keywords
        """
        if self.system_name == "Cardiovascular":
            return {
                "Ribcage (All Ribs)": {
                    'include': ['rib', 'first rib', 'second rib', 'third rib',
                                'fourth rib', 'fifth rib', 'sixth rib', 'seventh rib',
                                'eighth rib', 'ninth rib', 'tenth rib', 'eleventh rib',
                                'twelfth rib', 'costa'],
                    'exclude': []
                }
            }

        elif self.system_name == "Musculoskeletal":
            return {
                "All Leg Bones": {
                    'include': ['femur', 'tibia', 'fibula', 'patella', 'talus',
                                'calcaneus', 'metatarsal', 'phalanx', 'cuneiform',
                                'cuboid', 'navicular', 'sesamoid'],
                    'exclude': []
                },
                "Femur Only": {
                    'include': ['femur'],
                    'exclude': []
                },
                "Tibia & Fibula": {
                    'include': ['tibia', 'fibula'],
                    'exclude': []
                },
                "Foot Bones": {
                    'include': ['metatarsal', 'phalanx', 'cuneiform', 'cuboid',
                                'navicular', 'sesamoid', 'talus', 'calcaneus'],
                    'exclude': []
                }
            }

        elif self.system_name == "Nervous":
            return {
                "Entire Skull": {
                    # Skull bones always have "bone" or are atlas/axis
                    'include': ['bone', 'atlas', 'axis'],
                    'exclude': []  # Don't exclude anything - all bones should go
                },
                "Skull Cap (Top)": {
                    'include': ['frontal bone', 'parietal bone', 'occipital bone'],
                    'exclude': []
                },
                "Facial Bones": {
                    'include': ['zygomatic bone', 'maxilla', 'palatine bone', 'ethmoid'],
                    'exclude': []
                }
            }

        elif self.system_name == "Dental / Mouth":
            return {
                "Gums/Soft Tissue": {
                    'include': ['gum', 'gingiva', 'mucosa', 'tissue', 'soft'],
                    'exclude': []
                },
                "Jaw Bones": {
                    'include': ['mandible', 'maxilla', 'jaw'],
                    'exclude': ['tooth', 'teeth', 'incisor', 'canine', 'molar', 'premolar']
                }
            }

        else:
            return {}

    def identify_structures_in_group(self, group_config):
        """
        Identify which structures match the given group configuration
        Now supports CODE-BASED matching for precise identification

        Args:
            group_config: Dict with:
                - 'codes': List of code prefixes to INCLUDE (e.g., ['MM'] for skull)
                - 'exclude_codes': List of code prefixes to EXCLUDE (e.g., ['FJ'] for brain)
                - 'include': List of keyword strings (fallback/additional)
                - 'exclude': List of exclude keywords (fallback/additional)
                OR simple list of keywords (backward compatibility)

        Returns:
            List of matching surface dictionaries
        """
        matching = []

        # Handle backward compatibility - if it's a list, convert to dict
        if isinstance(group_config, list):
            include_keywords = group_config
            exclude_keywords = []
            include_codes = []
            exclude_codes = []
        else:
            include_keywords = group_config.get('include', [])
            exclude_keywords = group_config.get('exclude', [])
            include_codes = group_config.get('codes', [])
            exclude_codes = group_config.get('exclude_codes', [])

        for surf in self.current_surfaces_ref:
            name_lower = surf['name'].lower()
            name_upper = surf['name'].upper()  # For code matching

            # METHOD 1: CODE-BASED MATCHING (most precise)
            if include_codes:
                # Check if structure has any of the include codes
                has_include_code = any(
                    code.upper() in name_upper for code in include_codes)

                if has_include_code:
                    # Check if it should be excluded by code
                    has_exclude_code = any(
                        code.upper() in name_upper for code in exclude_codes)

                    if not has_exclude_code:
                        matching.append(surf)
                        continue  # Found by code, skip keyword matching

            # METHOD 2: KEYWORD-BASED MATCHING (fallback)
            if include_keywords:
                # Check if it matches include keywords
                has_include_match = any(
                    keyword in name_lower for keyword in include_keywords)

                if has_include_match:
                    # Then check if it should be excluded
                    has_exclude_match = any(
                        keyword in name_lower for keyword in exclude_keywords)

                    # Only add if included AND not excluded
                    if not has_exclude_match:
                        matching.append(surf)

        return matching

    # ==================== REMOVAL OPERATIONS ====================

    def remove_group(self, group_name, group_config):
        """
        PERMANENTLY remove a group of structures from the scene and current_surfaces list

        Args:
            group_name: Display name of the group
            group_config: Dict with 'include' and 'exclude' keywords OR list of keywords

        Returns:
            Number of structures removed
        """
        structures_to_remove = self.identify_structures_in_group(group_config)

        if not structures_to_remove:
            self.log_message(f"⚠️ No structures found matching '{group_name}'")
            return 0

        removed_count = 0

        for surf in structures_to_remove:
            # Remove actor from plotter
            if 'actor' in surf and surf['actor'] is not None:
                try:
                    self.plotter.remove_actor(surf['actor'])
                except Exception as e:
                    self.log_message(
                        f"⚠️ Could not remove actor for {surf['name']}: {e}")

            # Store for potential restore
            if surf not in self.removed_structures:
                self.removed_structures.append(surf)

            # CRITICAL: Remove from current_surfaces list
            if surf in self.current_surfaces_ref:
                self.current_surfaces_ref.remove(surf)
                removed_count += 1

        # Re-render the scene
        self.plotter.render()

        self.log_message(
            f"🗑️ PERMANENTLY removed {removed_count} structures from '{group_name}'")
        self.log_message(
            f"📊 Remaining structures: {len(self.current_surfaces_ref)}")
        return removed_count

    def restore_all(self):
        """
        Restore all removed structures back to the scene and current_surfaces list

        Returns:
            Number of structures restored
        """
        if not self.removed_structures:
            self.log_message("ℹ️ No structures to restore")
            return 0

        restored_count = 0

        for surf in self.removed_structures:
            # Re-add to current_surfaces
            if surf not in self.current_surfaces_ref:
                self.current_surfaces_ref.append(surf)

            # Re-add actor to plotter
            if 'mesh' in surf:
                try:
                    # Get stored opacity if available
                    opacity = 0.95
                    if 'actor' in surf and surf['actor'] is not None:
                        try:
                            opacity = surf['actor'].GetProperty().GetOpacity()
                        except:
                            pass

                    # Re-create actor
                    actor = self.plotter.add_mesh(
                        surf['mesh'],
                        color=surf.get('color', '#888888'),
                        opacity=opacity,
                        smooth_shading=True,
                        name=surf['name']
                    )
                    surf['actor'] = actor
                    restored_count += 1

                except Exception as e:
                    self.log_message(
                        f"⚠️ Could not restore {surf['name']}: {e}")

        # Clear the removed list
        self.removed_structures.clear()

        # Re-render
        self.plotter.render()

        self.log_message(f"✅ Restored {restored_count} structures")
        self.log_message(
            f"📊 Total structures now: {len(self.current_surfaces_ref)}")
        return restored_count

    # ==================== UI DIALOG ====================

    def show_removal_dialog(self, parent=None):
        """
        Show the selective removal control dialog
        """
        removal_groups = self.get_removal_groups()

        if not removal_groups:
            QtWidgets.QMessageBox.information(
                parent, "Not Available",
                f"Selective removal not configured for {self.system_name} system"
            )
            return

        # Create dialog
        self.removal_dialog = QtWidgets.QDialog(parent)
        self.removal_dialog.setWindowTitle(
            f"🗑️ Selective Removal - {self.system_name}")
        self.removal_dialog.setModal(False)
        self.removal_dialog.setMinimumWidth(550)
        self.removal_dialog.setMinimumHeight(400)

        # Styling
        self.removal_dialog.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                        stop:0 #2c3e50, stop:1 #34495e);
            }
            QGroupBox {
                background-color: #ecf0f1;
                border: 3px solid #e74c3c;
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
            QLabel {
                color: #2c3e50;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton#restoreButton {
                background-color: #27ae60;
            }
            QPushButton#restoreButton:hover {
                background-color: #229954;
            }
            QPushButton#closeButton {
                background-color: #7f8c8d;
            }
            QPushButton#closeButton:hover {
                background-color: #5d6d7e;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self.removal_dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QtWidgets.QLabel(f"🗑️ Selective Structure Removal")
        title.setStyleSheet(
            "color: #ecf0f1; font-size: 20px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QtWidgets.QLabel(
            f"Hide specific anatomical groups in {self.system_name} system"
        )
        subtitle.setStyleSheet(
            "color: #bdc3c7; font-size: 11px; font-style: italic;")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Scroll area for removal options
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)

        # Create removal buttons for each group
        for group_name, group_config in removal_groups.items():
            group_widget = self._create_removal_group_widget(
                group_name, group_config)
            scroll_layout.addWidget(group_widget)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Status label
        self.status_label = QtWidgets.QLabel("Ready to remove structures")
        self.status_label.setStyleSheet(
            "color: #ecf0f1; font-size: 11px; font-style: italic;")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Bottom buttons
        button_layout = QtWidgets.QHBoxLayout()

        btn_restore = QtWidgets.QPushButton("♻️ Restore All")
        btn_restore.setObjectName("restoreButton")
        btn_restore.setMinimumHeight(50)
        btn_restore.clicked.connect(self._on_restore_all)
        button_layout.addWidget(btn_restore)

        btn_close = QtWidgets.QPushButton("✅ Done")
        btn_close.setObjectName("closeButton")
        btn_close.setMinimumHeight(50)
        btn_close.clicked.connect(self.removal_dialog.close)
        button_layout.addWidget(btn_close)

        layout.addLayout(button_layout)

        # Show dialog
        self.removal_dialog.show()
        self.log_message("🗑️ Selective removal dialog opened")

    def _create_removal_group_widget(self, group_name, group_config):
        """Create a widget for one removal group"""
        group = QtWidgets.QGroupBox(f"🎯 {group_name}")
        layout = QtWidgets.QVBoxLayout()

        # Count structures in this group
        matching = self.identify_structures_in_group(group_config)
        count = len(matching)

        # Info label
        info_label = QtWidgets.QLabel(
            f"Found {count} structure(s) matching this group"
        )
        info_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        layout.addWidget(info_label)

        # Structure names preview (max 5)
        if matching:
            preview_names = [surf['name'] for surf in matching[:5]]
            preview_text = ", ".join(preview_names)
            if len(matching) > 5:
                preview_text += f" ... and {len(matching) - 5} more"

            preview_label = QtWidgets.QLabel(f"📋 {preview_text}")
            preview_label.setStyleSheet(
                "color: #95a5a6; font-size: 9px; font-style: italic;")
            preview_label.setWordWrap(True)
            layout.addWidget(preview_label)

        # Remove button
        btn_remove = QtWidgets.QPushButton(f"🗑️ Remove {group_name}")
        btn_remove.setMinimumHeight(45)
        btn_remove.clicked.connect(
            lambda: self._on_remove_group(group_name, group_config, count))

        if count == 0:
            btn_remove.setEnabled(False)
            btn_remove.setText(f"⚠️ No {group_name} Found")

        layout.addWidget(btn_remove)

        group.setLayout(layout)
        return group

    def _on_remove_group(self, group_name, group_config, expected_count):
        """Handle removal button click"""
        removed = self.remove_group(group_name, group_config)

        if removed > 0:
            self.status_label.setText(
                f"✅ Removed {removed}/{expected_count} from '{group_name}'"
            )
            self.status_label.setStyleSheet(
                "color: #2ecc71; font-size: 11px; font-weight: bold;")
        else:
            self.status_label.setText(f"⚠️ Could not remove '{group_name}'")
            self.status_label.setStyleSheet(
                "color: #e74c3c; font-size: 11px; font-weight: bold;")

    def _on_restore_all(self):
        """Handle restore all button click"""
        restored = self.restore_all()

        if restored > 0:
            self.status_label.setText(f"♻️ Restored {restored} structures")
            self.status_label.setStyleSheet(
                "color: #3498db; font-size: 11px; font-weight: bold;")
        else:
            self.status_label.setText("ℹ️ No structures to restore")
            self.status_label.setStyleSheet(
                "color: #95a5a6; font-size: 11px; font-style: italic;")
