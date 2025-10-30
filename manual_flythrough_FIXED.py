"""
Manual Fly-through Feature
Interactive path drawing by clicking on 3D anatomy
Camera flies through the drawn path smoothly
"""

import numpy as np
import pyvista as pv
from PyQt5 import QtWidgets, QtCore
try:
    from scipy.interpolate import CatmullRomSpline
    HAS_CATMULL_ROM = True
except ImportError:
    from scipy.interpolate import interp1d
    HAS_CATMULL_ROM = False


class ManualFlythroughController:
    """Controller for manual path drawing and fly-through animation"""

    def __init__(self, plotter, meshes, console_log=None):
        self.plotter = plotter
        self.meshes = meshes  # List of PyVista meshes in the scene
        self.log_message = console_log if console_log else print

        # Path data
        self.waypoints = []  # List of 3D points clicked by user
        self.waypoint_actors = []  # Visual markers for waypoints
        self.path_line_actor = None  # Line connecting waypoints
        self.smooth_path = None  # Smoothed spline path for animation

        # Drawing state
        self.is_drawing_mode = False
        self.click_observer = None

        # Animation state
        self.is_animating = False
        self.animation_timer = None
        self.current_frame = 0
        self.total_frames = 200
        self.camera_path = []

    def enter_drawing_mode(self):
        """Enable interactive path drawing mode"""
        self.log_message("\n🎨 MANUAL PATH DRAWING MODE")
        self.log_message("=" * 60)
        self.log_message("📍 CTRL + Left Click on anatomy to place waypoints")
        self.log_message("📍 Right Click to finish drawing")
        self.log_message("=" * 60)

        self.is_drawing_mode = True
        self.waypoints = []
        self.clear_visual_markers()

        # Add click observer
        self.click_observer = self.plotter.iren.add_observer(
            'LeftButtonPressEvent', self.on_click
        )

        # Add right-click observer to finish
        self.right_click_observer = self.plotter.iren.add_observer(
            'RightButtonPressEvent', self.finish_drawing
        )

    def on_click(self, obj, event):
        """Handle click event - pick point on mesh"""
        if not self.is_drawing_mode:
            return

        # Check if CTRL is pressed
        interactor = self.plotter.iren.interactor
        if not interactor.GetControlKey():
            return

        # Get click position
        click_pos = interactor.GetEventPosition()

        # Perform ray picking
        picker = self.plotter.iren.picker
        picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)

        picked_position = picker.GetPickPosition()

        # Check if we actually hit something
        if picker.GetMapper() is not None:
            # Valid 3D point picked
            point_3d = np.array(picked_position)
            self.waypoints.append(point_3d)

            self.log_message(f"✅ Waypoint #{len(self.waypoints)}: {point_3d}")

            # Add visual marker
            self.add_waypoint_marker(point_3d, len(self.waypoints))

            # Update path line
            if len(self.waypoints) >= 2:
                self.update_path_line()

            self.plotter.render()

    def add_waypoint_marker(self, position, number):
        """Add a visible sphere marker at waypoint"""
        sphere = pv.Sphere(radius=2.0, center=position)

        # Color based on sequence
        colors = ['#FF0000', '#FF6600', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#FF00FF']
        color = colors[(number - 1) % len(colors)]

        actor = self.plotter.add_mesh(
            sphere,
            color=color,
            opacity=0.9,
            render=False
        )

        self.waypoint_actors.append(actor)


    def update_path_line(self):
        """Draw line connecting all waypoints"""
        if self.path_line_actor is not None:
            self.plotter.remove_actor(self.path_line_actor)

        if len(self.waypoints) < 2:
            return

        # Create polyline through waypoints
        points = np.array(self.waypoints)
        line = pv.Spline(points, n_points=len(points) * 10)

        self.path_line_actor = self.plotter.add_mesh(
            line,
            color='cyan',
            line_width=4,
            opacity=0.8,
            render=False
        )

    def finish_drawing(self, obj, event):
        """User finished drawing path"""
        if not self.is_drawing_mode:
            return

        self.is_drawing_mode = False

        # Remove observers
        if self.click_observer:
            self.plotter.iren.remove_observer(self.click_observer)
        if self.right_click_observer:
            self.plotter.iren.remove_observer(self.right_click_observer)

        if len(self.waypoints) < 2:
            self.log_message("⚠️ Need at least 2 waypoints to create a path")
            self.clear_visual_markers()
            return

        self.log_message(f"\n✅ Path completed with {len(self.waypoints)} waypoints")
        self.log_message("🎬 Ready to animate! Click 'Start Animation'")

        # Generate smooth path
        self.generate_smooth_path()

    def generate_smooth_path(self):
        """Create smooth spline through waypoints"""
        if len(self.waypoints) < 2:
            return False

        points = np.array(self.waypoints)

        # Create smooth Catmull-Rom spline
        try:
            # Parameter for each waypoint
            t = np.linspace(0, 1, len(points))

            # Create spline for each dimension
            cs_x = CatmullRomSpline(t, points[:, 0])
            cs_y = CatmullRomSpline(t, points[:, 1])
            cs_z = CatmullRomSpline(t, points[:, 2])

            # Sample the spline at many points
            t_smooth = np.linspace(0, 1, self.total_frames)

            smooth_x = cs_x(t_smooth)
            smooth_y = cs_y(t_smooth)
            smooth_z = cs_z(t_smooth)

            self.smooth_path = np.column_stack([smooth_x, smooth_y, smooth_z])

            self.log_message(f"✅ Smooth path generated: {len(self.smooth_path)} frames")
            return True

        except Exception as e:
            self.log_message(f"⚠️ Spline generation failed: {e}")
            # Fallback: linear interpolation
            self.smooth_path = self.linear_interpolate_path()
            return True

    def linear_interpolate_path(self):
        """Fallback: simple linear interpolation between waypoints"""
        points = np.array(self.waypoints)
        path = []

        frames_per_segment = self.total_frames // (len(points) - 1)

        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            for t in np.linspace(0, 1, frames_per_segment, endpoint=False):
                pos = start * (1 - t) + end * t
                path.append(pos)

        path.append(points[-1])  # Add final point

        return np.array(path)

    def start_animation(self, speed=50):
        """Start camera animation along the path"""
        if self.smooth_path is None or len(self.smooth_path) < 2:
            self.log_message("⚠️ No path to animate. Draw a path first!")
            return False

        if self.is_animating:
            self.log_message("⚠️ Animation already running")
            return False

        self.log_message("\n🎬 STARTING FLY-THROUGH ANIMATION")
        self.log_message("=" * 60)

        self.is_animating = True
        self.current_frame = 0

        # Hide path visualization during animation
        self.hide_path_markers()

        # Create timer
        self.animation_timer = QtCore.QTimer()
        self.animation_timer.timeout.connect(self.animate_frame)
        self.animation_timer.start(speed)  # milliseconds per frame

        return True

    def animate_frame(self):
        """Update camera position for current frame"""
        if not self.is_animating or self.smooth_path is None:
            return

        if self.current_frame >= len(self.smooth_path) - 1:
            self.stop_animation()
            return

        # Current and next positions
        current_pos = self.smooth_path[self.current_frame]

        # Look ahead for focal point
        look_ahead = min(5, len(self.smooth_path) - self.current_frame - 1)
        focal_pos = self.smooth_path[self.current_frame + look_ahead]

        # Set camera
        camera = self.plotter.camera
        camera.position = current_pos
        camera.focal_point = focal_pos

        # Calculate up vector (banking effect)
        if self.current_frame > 0 and self.current_frame < len(self.smooth_path) - 1:
            prev_pos = self.smooth_path[self.current_frame - 1]
            next_pos = self.smooth_path[self.current_frame + 1]

            # Direction vectors
            direction = focal_pos - current_pos
            tangent = next_pos - prev_pos

            # Cross product for banking
            if np.linalg.norm(tangent) > 0.001:
                tangent = tangent / np.linalg.norm(tangent)
                default_up = np.array([0, 0, 1])

                # Smooth banking
                up = np.cross(np.cross(tangent, default_up), tangent)
                if np.linalg.norm(up) > 0.001:
                    camera.up = up / np.linalg.norm(up)

        self.plotter.render()
        self.current_frame += 1

        # Progress logging
        if self.current_frame % 20 == 0:
            progress = (self.current_frame / len(self.smooth_path)) * 100
            self.log_message(f"   📍 Progress: {progress:.1f}%")

    def stop_animation(self):
        """Stop the animation"""
        if self.animation_timer:
            self.animation_timer.stop()

        self.is_animating = False
        self.current_frame = 0

        # Show path markers again
        self.show_path_markers()

        self.log_message("\n⏹️ Animation stopped")
        self.log_message("✅ Path markers restored")

    def hide_path_markers(self):
        """Hide waypoint markers and path line during animation"""
        for actor in self.waypoint_actors:
            actor.SetVisibility(False)

        if self.path_line_actor:
            self.path_line_actor.SetVisibility(False)

        self.plotter.render()

    def show_path_markers(self):
        """Show waypoint markers and path line"""
        for actor in self.waypoint_actors:
            actor.SetVisibility(True)

        if self.path_line_actor:
            self.path_line_actor.SetVisibility(True)

        self.plotter.render()

    def clear_visual_markers(self):
        """Remove all visual markers"""
        for actor in self.waypoint_actors:
            self.plotter.remove_actor(actor)

        if self.path_line_actor:
            self.plotter.remove_actor(self.path_line_actor)

        self.waypoint_actors = []
        self.path_line_actor = None
        self.plotter.render()

    def reset(self):
        """Reset everything"""
        self.stop_animation()
        self.clear_visual_markers()
        self.waypoints = []
        self.smooth_path = None
        self.is_drawing_mode = False

        self.log_message("🔄 Manual fly-through reset")