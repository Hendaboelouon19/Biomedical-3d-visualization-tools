import numpy as np
import nibabel as nib
import pyvista as pv
from skimage import measure

# نفس ال label maps اللي اتفقنا عليها
LABEL_NAMES = {
    205: "Left Ventricle",
    420: "Myocardium",
    500: "Right Ventricle",
    550: "Left Atrium",
    600: "Right Atrium",
    820: "Aorta",
    850: "Pulmonary Artery",
}

COLOR_MAP = {
    205: [0.9, 0.3, 0.3],   # LV cavity      - أحمر لحمي
    500: [0.3, 0.4, 1.0],   # RV cavity      - أزرق
    550: [1.0, 0.9, 0.3],   # LA             - أصفر
    600: [0.4, 1.0, 0.4],   # RA             - أخضر
    420: [1.0, 0.6, 0.7],   # Myocardium     - وردي/لحم
    820: [1.0, 0.6, 0.2],   # Aorta          - برتقالي
    850: [0.7, 0.2, 1.0],   # Pulmonary Art. - بنفسجي
}


def _marching_cubes_single_label(seg_data, affine, label_value, min_voxels=500):
    """
    seg_data: segmentation volume (3D numpy)
    affine:   affine matrix from nib
    label_value: which structure to extract
    returns: pv.PolyData mesh (smoothed) or None
    """
    # binary mask of this structure
    mask = (seg_data == label_value).astype(np.uint8)

    # skip ultra tiny blobs
    if np.sum(mask) < min_voxels:
        return None

    # marching cubes on the binary mask
    verts, faces, normals, values = measure.marching_cubes(
        volume=mask,
        level=0.5,
        spacing=tuple(abs(v) for v in np.diag(affine[:3, :3]))
    )

    # apply affine to move from voxel index -> world coords
    verts_h = np.column_stack([verts, np.ones((verts.shape[0], 1))])  # (N,4)
    verts_world = verts_h @ affine.T
    verts_world = verts_world[:, :3]

    # pyvista wants [3, i, j, k, 3, i, j, k, ...]
    faces_pv = []
    for tri in faces:
        faces_pv.extend([3, int(tri[0]), int(tri[1]), int(tri[2])])
    faces_pv = np.array(faces_pv)

    mesh = pv.PolyData(verts_world, faces_pv)

    # smooth surface so it looks organic (not voxel blocky)
    mesh = mesh.smooth(
        n_iter=60,
        relaxation_factor=0.1,
        feature_smoothing=False,
        boundary_smoothing=True
    )

    return mesh


def build_heart_surfaces_from_seg(seg_path, console_log=lambda msg: None):
    """
    High-level:
    - load NIfTI seg
    - loop on all labels
    - build pv.PolyData mesh for each label
    - return list of dicts: { 'name', 'color', 'mesh' }

    GUI will take that list and add meshes to its plotter.
    """
    console_log(f"[INFO] Loading segmentation: {seg_path}")
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata()
    affine = seg_nii.affine

    console_log(f"[INFO] seg shape: {seg_data.shape}")

    labels_present = np.unique(seg_data)
    labels_present = [int(v) for v in labels_present if v != 0]
    console_log(f"[INFO] labels found: {labels_present}")

    surfaces = []

    for label_value in labels_present:
        struct_name = LABEL_NAMES.get(label_value, f"Structure {label_value}")
        struct_color = COLOR_MAP.get(label_value, [0.8, 0.8, 0.8])

        console_log(f"[BUILD] {label_value} → {struct_name}")

        mesh = _marching_cubes_single_label(seg_data, affine, label_value)
        if mesh is None:
            console_log(f"[SKIP] {struct_name} is too small / empty")
            continue

        surfaces.append({
            "name": struct_name,
            "color": struct_color,
            "mesh": mesh,
        })

    console_log("[INFO] Finished surface extraction.")
    return surfaces

