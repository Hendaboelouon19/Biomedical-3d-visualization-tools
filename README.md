

**Repository name:** `Biomedical-3D-MPR-Visualization`  
**Description:**  
An interactive biomedical visualization platform combining **3D anatomy rendering**, **segmentation-based navigation**, **multi-planar reconstruction (MPR)**, **curved reconstruction**, and **physiological motion simulation** — designed for real-time exploration of medical imaging data (CT/MRI/NIfTI) across multiple body systems.

---

## 📂 Repository Structure
Biomedical-3D-MPR-Visualization/
│
├── data/
│ ├── ct_train/ # CT/MRI datasets (.nii, .nii.gz, .mha)
│ ├── segmentation_results/ # Output from AI segmentation (heart, brain, teeth, leg)
│ ├── models/ # Pretrained models for organ/orientation detection
│ └── examples/ # Example cases used in the video demo
│
├── src/
│ ├── gui_last.py # Main GUI connecting all features
│ ├── surface_rendering.py # Feature 1: Show Anatomy
│ ├── focus_navigation.py # Feature 2: Focus Navigation
│ ├── clipping_planes.py # Feature 3: Clipping Planes + MPR Viewer
│ ├── curved_mpr.py # Feature 4: Curved MPR
│ ├── flythrough.py # Feature 5: Fly-Through Navigation
│ ├── moving_systems.py # Feature 6: Moving Systems Simulation
│ ├── utils/ # Helper functions (data loading, transformations)
│ └── requirements.txt # Dependencies
│
├── results/
│ ├── screenshots/ # Static output previews
│ └── demo_video.mp4 # Task demonstration video
│
└── README.md


---

## 🧩 Systems and Organs Overview

| System | Organs Included | Notes |
|--------|------------------|-------|
| **Nervous System** | Brain | Includes MPR, signal path animation, fly-through |
| **Cardiovascular System** | Heart | Includes pumping + blood flow animation |
| **Musculoskeletal System** | Leg | Full clipping, curved MPR, and focus navigation |
| **Dental System** | Teeth | Curved MPR reconstruction and clipping for arches |

> All features (1–5) are applied to every organ except **Feature 6**, which applies only to **Heart** and **Brain**.

---

## ⚙️ Features and Implementation

### 🩻 **Feature 1 — Surface Rendering (Show Anatomy)**
**File:** `surface_rendering.py`  
**Goal:** Display 3D anatomy of organs with layered visibility control.

**Core Logic**
- Loads volumetric meshes using `pyvista.read()` or `vtk.vtkPolyDataMapper()`
- Applies medical color maps (skin tone, bone white, muscle pink, artery red, vein blue)
- User toggles visibility between:
  - Skull, bones, ribs  
  - Organs (brain, heart, leg, teeth)

```python
actor.GetProperty().SetOpacity(1.0 if visible else 0.0)

🎯 Feature 2 — Focus Navigation

File: focus_navigation.py
Goal: Focus camera and opacity on a specific organ region from a dropdown.

Logic

selected_part = self.dropdown.currentText()
target_actor = self.find_actor(selected_part)
for actor in self.actors:
    actor.GetProperty().SetOpacity(0.1)
target_actor.GetProperty().SetOpacity(1.0)
self.camera.SetFocalPoint(target_actor_center)


Works using segmentation masks loaded with nibabel.load()

Smooth camera interpolation and re-focusing

✂️ Feature 3 — Clipping Planes (with Integrated MPR Viewer)

File: clipping_planes.py
Includes 3 modes

3D Volumetric Mode – cuts raw 3D CT/MRI volumes using VTK clipping planes

NIfTI Mode – performs clipping and overlays the 2D segmentation slice

MPR Viewer Mode – synchronized perpendicular planes (axial, coronal, sagittal) with real-time movement

plane = vtk.vtkPlane()
plane.SetOrigin(x, y, z)
plane.SetNormal(nx, ny, nz)
clipper = vtk.vtkClipDataSet()
clipper.SetClipFunction(plane)
clipper.SetInputData(volume_data)


MPR Integration

self.mpr_viewer.update_slices(x_idx, y_idx, z_idx)


Displays perpendicular planes dynamically with perfect alignment (see demo images).

🌀 Feature 4 — Curved MPR

File: curved_mpr.py
Goal: Flatten curved anatomical structures for clear visualization (vessels, dental arches, etc.)

Logic

User selects multiple points on a curve

The app computes normals and extracts perpendicular slices

Concatenates slices into a straight continuous line

reslicer.SetResliceAxesDirectionCosines(normal, tangent, cross)


Output: Straightened image of curved anatomy.

🚀 Feature 5 — Fly-Through Navigation

File: flythrough.py
Modes

Automatic – system selects an internal organ path

Manual Path – user draws 3D points; camera follows trajectory

Predefined Path – uses biological pathways (heart blood flow, brain signal path)

for p in path_points:
    self.camera.SetPosition(p)
    self.camera.SetFocalPoint(next_point)
    self.render_window.Render()


Dropdown includes:

Automatic fly-through

Manual drawing

Default heart/brain paths

❤️ Feature 6 — Moving Systems Simulation

File: moving_systems.py
Goal: Add physiological motion simulation.

Heart: Beats synchronized with oxygenated/deoxygenated blood stream animation

scale = 1.0 + 0.05 * np.sin(time * 2 * np.pi * bpm / 60)
actor.SetScale(scale)


Brain: Signal propagation animation for

Seeing 👁️

Thinking 💭

Heart Control ❤️
Each pathway glows dynamically to indicate neural activity.

🖥️ GUI Integration

File: gui_last.py
Description: Main GUI linking all visualization features.

Tabs
Nervous System
Cardiovascular System
Musculoskeletal System
Dental System

Buttons

Load NIfTI
Show Anatomy
Focus on Part
Clipping Plane
Curved MPR
Fly Through
Reset View
Each tab dynamically loads the corresponding dataset and reuses the visualization pipeline.
🎥 Video Demonstration
file: results/demo_video.mp4
Showcases
Loading CT/NIfTI datasets
Surface rendering of skull → hiding bones → showing brain
Focus on segmented brain region
3D clipping + synchronized MPR viewer
Curved MPR reconstruction
Fly-through (auto/manual/default)
Heart pumping and brain signal glow
System tab switching (brain, heart, leg, teeth)
Resetting 3D view
▶️ How to Run
# Clone repository
git clone https://github.com/<your-username>/Biomedical-3D-MPR-Visualization.git
cd Biomedical-3D-MPR-Visualization/src
# Install requirements
pip install -r requirements.txt
# Launch GUI
python gui_last.py
🧠 Summary

This platform unifies AI segmentation, 3D reconstruction, and interactive anatomical exploration into one powerful tool.
It allows seamless transitions between 2D/3D/MPR views, precise focusing, and dynamic physiological simulation — making it ideal for medical education, research, and clinical visualization.
