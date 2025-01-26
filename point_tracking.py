import bpy
import torch
import cv2
import os
from tqdm import tqdm

from bpy.types import Panel, Operator, PropertyGroup

from .models import build_point_tracker
from .utils import get_vars_from_context

# ---------------------------------------------------------------------------
#  Property Group & Global Properties
# ---------------------------------------------------------------------------


class SelectedTrackItem(PropertyGroup):
    """Represents a single selected track (by name)."""

    name: bpy.props.StringProperty(name="Track Name")


RESOLUTION_OPTIONS = {
    "256x256": (256, 256),
    "512x512": (512, 512),
}


def init_properties():
    """
    Register the addon-scoped properties on bpy.types.Scene.
    """

    # Display string showing the number of selected tracks.
    bpy.types.Scene.selected_tracks_display = bpy.props.StringProperty(
        name="Selected Tracks Display",
        description="Display the number of selected tracks",
        default="No tracks selected",
    )

    # A collection of selected tracks, by name.
    bpy.types.Scene.selected_tracks = bpy.props.CollectionProperty(
        type=SelectedTrackItem
    )

    # CPU vs GPU
    bpy.types.Scene.device = bpy.props.EnumProperty(
        name="Compute Mode",
        description="Select CPU or GPU",
        items=[
            ("GPU", "GPU", ""),
            ("CPU", "CPU", ""),
        ],
        default="GPU",
    )

    # Online vs Offline
    bpy.types.Scene.mode = bpy.props.EnumProperty(
        name="Compute Mode",
        description="Online vs. Offline Tracking",
        items=[
            ("Online", "Online", ""),
            ("Offline", "Offline", ""),
        ],
        default="Online",
    )

    # Resolution enumerator
    bpy.types.Scene.resolution = bpy.props.EnumProperty(
        name="Resolution",
        description="Select the desired resolution",
        items=[
            ("256x256", "Low Res", "Set resolution to 256x256 pixels"),
            ("512x512", "High Res", "Set resolution to 512x512 pixels"),
        ],
        default="256x256",
    )


def clear_properties():
    """Clean up the properties when the addon is unregistered."""
    del bpy.types.Scene.selected_tracks_display
    del bpy.types.Scene.selected_tracks
    del bpy.types.Scene.device
    del bpy.types.Scene.mode
    del bpy.types.Scene.resolution


# ---------------------------------------------------------------------------
#  UI Panel
# ---------------------------------------------------------------------------


class MotionTrackingPanel(Panel):
    bl_idname = "CLIP_PT_point_tracking_panel"
    bl_label = "PointsTracker"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "PointsTracker"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # ----- Tracks Selection Section -----
        layout.label(text="Tracks Selection:")
        row = layout.row()
        row.operator("point_tracking.select_tracks", text="Select Tracks")
        row = layout.row()
        row.operator(
            "point_tracking.select_isolated_tracks", text="Select Isolated Tracks"
        )

        # Display how many tracks are selected (updated by the operators).
        row = layout.row()
        row.label(text=scene.selected_tracks_display, icon="TRACKING")

        layout.separator()
        layout.label(text="Model Settings:")

        # CPU/GPU selection
        row = layout.row()
        row.prop(scene, "device", expand=True)

        # Online/Offline selection
        # row = layout.row()
        # row.prop(scene, "mode", expand=True)

        # Resolution selection
        row = layout.row()
        row.prop(scene, "resolution", expand=True)

        # Start tracking
        layout.label(text="Start Tracking:")
        row = layout.row()
        row.operator("point_tracking.start_tracking", text="Go !")


# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------


class SelectTracks(Operator):
    bl_idname = "point_tracking.select_tracks"
    bl_label = "Select Tracks"

    def execute(self, context):
        """
        Gathers all currently selected tracks in the Movie Clip editor,
        stores their names in 'context.scene.selected_tracks',
        and updates the 'selected_tracks_display' property.
        """
        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected.")
            context.scene.selected_tracks_display = "No movie clip selected."
            return {"CANCELLED"}

        # Gather selected tracks
        selected_tracks = [t for t in clip.tracking.tracks if t.select]

        # Clear any old data
        context.scene.selected_tracks.clear()

        # Add newly selected track names
        for track in selected_tracks:
            item = context.scene.selected_tracks.add()
            item.name = track.name

        # Update display property
        num_selected = len(selected_tracks)
        context.scene.selected_tracks_display = (
            f"{num_selected} track{'s' if num_selected != 1 else ''} selected"
            if num_selected > 0
            else "No tracks selected"
        )

        self.report(
            {"INFO"},
            f"{num_selected} track{'s' if num_selected != 1 else ''} selected.",
        )
        return {"FINISHED"}


class SelectIsolatedTracks(Operator):
    bl_idname = "point_tracking.select_isolated_tracks"
    bl_label = "Select Isolated Tracks"
    bl_description = "Select tracking tracks that have only one marker."

    def execute(self, context):
        """
        Identifies “isolated” tracks (tracks with only one marker),
        selects them, updates the scene’s selected_tracks,
        and updates the display property accordingly.
        """
        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected.")
            context.scene.selected_tracks_display = "No movie clip selected."
            return {"CANCELLED"}

        # Gather isolated tracks (only one marker)
        isolated_tracks = [t for t in clip.tracking.tracks if len(t.markers) == 1]

        # Clear existing selection in Blender
        for track in clip.tracking.tracks:
            track.select = False

        # Select the isolated tracks
        for track in isolated_tracks:
            track.select = True

        # Clear old data in property group
        context.scene.selected_tracks.clear()

        # Add newly selected isolated track names
        for track in isolated_tracks:
            item = context.scene.selected_tracks.add()
            item.name = track.name

        # Update display property
        num_selected = len(isolated_tracks)
        context.scene.selected_tracks_display = (
            f"{num_selected} isolated track{'s' if num_selected != 1 else ''} selected"
            if num_selected > 0
            else "No isolated tracks found"
        )

        self.report(
            {"INFO"},
            f"{num_selected} isolated track{'s' if num_selected != 1 else ''} selected.",
        )
        return {"FINISHED"}


class StartTracking(Operator):
    bl_idname = "point_tracking.start_tracking"
    bl_label = "Start Tracking"
    bl_options = {"REGISTER", "UNDO"}

    def __init__(self):
        super().__init__()
        self.tracker_type = "tapir"
        self.frames = None
        self.current_tracks = None
        self.device = None
        self.frame_index = 0
        self.start_frame = 0
        self.total_frames = 0

    def _build_model(self, context):
        """
        Build and move the tracker to the selected device (CPU/GPU),
        taking into account GPU availability (CUDA, MPS, fallback to CPU).
        """
        selected_device = context.scene.device
        if selected_device == "GPU":
            # Try CUDA -> MPS -> CPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
                self.report({"WARNING"}, "GPU not available, defaulting to CPU mode.")
        else:
            device = torch.device("cpu")

        try:
            self.tracker = build_point_tracker(
                tracker_type=self.tracker_type,
                mode=context.scene.mode,
                resolution=RESOLUTION_OPTIONS[context.scene.resolution],
                device=device,
            )
            self.tracker.to(device)
            self.device = device
            return True
        except Exception as e:
            self.report({"ERROR"}, f"Failed to build tracker: {e}")
            return False

    def _retrieve_selected_tracks(self, context):
        """
        Get the track objects from the Blender clip,
        matching the names stored in 'context.scene.selected_tracks'.
        """
        selected_track_names = [item.name for item in context.scene.selected_tracks]
        if not selected_track_names:
            self.report({"ERROR"}, "No tracks selected. Please select tracks first.")
            return None

        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected.")
            return None

        # Filter the tracks by those that match the stored names
        selected_tracks = [
            t for t in clip.tracking.tracks if t.name in selected_track_names
        ]
        if not selected_tracks:
            self.report({"ERROR"}, "Selected tracks not found in the current clip.")
            return None

        return selected_tracks

    def _extract_query_tracks(self, tracks):
        """
        For each track, get the first marker’s frame index + (x, y) position.
        Returns a (num_tracks x 3) tensor: [frame_idx, x, y].
        """
        try:
            first_markers = [track.markers[0] for track in tracks]
        except IndexError:
            self.report({"ERROR"}, "One or more tracks do not have any markers.")
            return None

        frames = torch.tensor([m.frame for m in first_markers], dtype=torch.int32)
        co_tensor = torch.stack(
            [torch.tensor(m.co, dtype=torch.float32) for m in first_markers],
            dim=0,
        )
        return torch.cat((frames.unsqueeze(1), co_tensor), dim=1)

    def _get_video_frames(self, clip):
        """
        Load frames from a MovieClip, handling both MOVIE (single file)
        and SEQUENCE (multiple image files). Returns a list of frames
        as NumPy arrays (BGR) via OpenCV.
        """
        source_type = clip.source
        frames = []

        if source_type == "MOVIE":
            # Single video file
            video_path = bpy.path.abspath(clip.filepath)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.report({"ERROR"}, f"Unable to open movie file: {video_path}")
                return None

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    self.report({"ERROR"}, f"Failed to read frame {i} from video.")
                    break
                frames.append(frame)
            cap.release()

        elif source_type == "SEQUENCE":
            # Image sequence
            directory = os.path.dirname(bpy.path.abspath(clip.filepath))
            for f in clip.files:
                seq_filepath = os.path.join(directory, f.name)
                frame = cv2.imread(seq_filepath)
                if frame is None:
                    self.report({"ERROR"}, f"Failed to read {seq_filepath}")
                    break
                frames.append(frame)

        else:
            self.report({"ERROR"}, f"Unsupported clip source type: {source_type}")
            return None

        return frames

    @torch.no_grad()
    def _online_tracking_init(self, context):
        """
        Initialize tracker in 'Online' mode with the query marker positions.
        Adjust frames so that the earliest marker’s frame is the start.
        """
        query_data = self._extract_query_tracks(self.current_tracks)
        if query_data is None:
            return False

        # The earliest frame among the selected tracks
        self.start_frame = int(query_data[:, 0].min().item())

        # Slice frames from start_frame - 1 so that frame indices match model input
        self.frames = self.frames[self.start_frame - 1 :]
        # Shift query_data frame numbers so the earliest frame becomes 0
        query_data[:, 0] -= self.start_frame

        # Initialize the tracker
        self.tracker.init_tracker(self.frames, query_data)
        return True

    @torch.no_grad()
    def _online_tracking_step(self, context):
        """
        Process frames sequentially and insert new markers in Blender
        if predictions are within valid range.
        """
        if self.frame_index >= len(self.frames):
            return False  # End of frames

        current_frame_global = self.frame_index + self.start_frame
        self.report(
            {"INFO"},
            f"Tracking frame {current_frame_global}/"
            f"{self.start_frame + self.total_frames - 1}",
        )

        frame = self.frames[self.frame_index]
        pred_tracks, pred_visibility = self.tracker.predict_frame([frame])

        # Insert markers in Blender for each visible track
        for i in range(len(pred_tracks)):
            if pred_visibility[i]:
                coord = pred_tracks[i].cpu().numpy()
                # Flip Y to match Blender’s coordinate system
                coord[1] = 1.0 - coord[1]

                if 0 <= coord[0] <= 1 and 0 <= coord[1] <= 1:
                    marker = self.current_tracks[i].markers.insert_frame(
                        current_frame_global
                    )
                    marker.co = coord
                else:
                    print(
                        "Predicted coordinates outside [0,1]. Skipping marker insertion."
                    )

        # Update Blender’s timeline and clip editor
        context.scene.frame_current = current_frame_global
        for area in context.screen.areas:
            if area.type == "CLIP_EDITOR":
                for space in area.spaces:
                    if space.type == "CLIP_EDITOR" and space.clip_user is not None:
                        space.clip_user.frame_current = current_frame_global

        # Update progress bar
        context.window_manager.progress_update(current_frame_global)

        self.frame_index += 1
        return True

    @torch.no_grad()
    def _offline_tracking(self, frames, current_tracks):
        """
        Stub for offline tracking logic (not implemented).
        """
        raise NotImplementedError("Offline tracking not yet implemented.")

    def invoke(self, context, event):
        """
        Initializes the tracker model, loads frames from the clip, retrieves selected tracks,
        and sets up for the modal operation if 'Online' mode is chosen.
        """
        if not self._build_model(context):
            return {"CANCELLED"}

        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected.")
            return {"CANCELLED"}

        self.frames = self._get_video_frames(clip)
        if self.frames is None:
            return {"CANCELLED"}

        self.total_frames = len(self.frames)
        self.current_tracks = self._retrieve_selected_tracks(context)
        if self.current_tracks is None:
            return {"CANCELLED"}

        if context.scene.mode == "Online":
            if not self._online_tracking_init(context):
                return {"CANCELLED"}
        else:
            raise NotImplementedError(
                "Offline tracking not yet implemented in this script."
            )

        # Start progress bar
        context.window_manager.progress_begin(self.start_frame, self.total_frames)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        """
        Modal loop for tracking in 'Online' mode. Cancels on ESC or right-click.
        """
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self.report({"WARNING"}, "Tracking canceled by user.")
            context.window_manager.progress_end()
            return {"CANCELLED"}

        if context.scene.mode == "Online":
            running = self._online_tracking_step(context)
            if not running:
                self.report({"INFO"}, "Tracking completed successfully.")
                context.window_manager.progress_end()
                return {"FINISHED"}
        else:
            raise NotImplementedError("Offline modal stepping not yet implemented.")

        return {"RUNNING_MODAL"}


# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------

classes = (
    SelectedTrackItem,
    MotionTrackingPanel,
    SelectTracks,
    SelectIsolatedTracks,
    StartTracking,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    init_properties()


def unregister():
    clear_properties()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
