import bpy
import torch
import cv2
from tqdm import tqdm

from bpy.types import Panel, Operator, PropertyGroup

from .models import build_point_tracker
from .utils import get_vars_from_context


# Define a PropertyGroup to store track names
class SelectedTrackItem(PropertyGroup):
    name: bpy.props.StringProperty(name="Track Name")


# Define a mapping from string identifiers to resolution tuples
RESOLUTION_OPTIONS = {
    "256x256": (256, 256),
    "512x512": (512, 512),
}


def init_properties():
    bpy.types.Scene.selected_points_display = bpy.props.StringProperty(
        name="Selected Points Display",
        description="Display the number of selected points",
        default="",
    )

    bpy.types.Scene.selected_tracks = bpy.props.CollectionProperty(
        type=SelectedTrackItem
    )

    bpy.types.Scene.device = bpy.props.EnumProperty(
        name="Compute Mode",
        description="Select CPU or GPU",
        items=[
            ("GPU", "GPU", ""),
            ("CPU", "CPU", ""),
        ],
        default="GPU",
    )

    bpy.types.Scene.mode = bpy.props.EnumProperty(
        name="Compute Mode",
        description="Select CPU or GPU",
        items=[
            ("Online", "Online", ""),
            ("Offline", "Offline", ""),
        ],
        default="Online",
    )
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
    del bpy.types.Scene.selected_points_display
    del bpy.types.Scene.selected_tracks


class MotionTrackingPanel(Panel):
    bl_idname = "CLIP_PT_point_tracking_panel"
    bl_label = "Point Tracking"
    bl_space_type = "CLIP_EDITOR"
    bl_region_type = "UI"
    bl_category = "Point Tracking"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # Section for selecting points
        layout.label(text="Tracks Selection")
        row = layout.row()
        row.operator("point_tracking.select_points", text="Select Tracks")
        row = layout.row()
        row.operator(
            "point_tracking.select_isolated_points", text="Select Isolated Tracks"
        )  # New Button

        # Display selected points
        layout.label(text="Selected Tracks:")
        row = layout.row()
        row.prop(scene, "selected_points_display", text="")

        ## Settings
        layout.separator()
        layout.label(text="Model Settings :")

        # CPU/GPU/MPS Selector
        row = layout.row()
        row.prop(scene, "device", expand=True)

        # Online/Offline Selector
        # row = layout.row()
        # row.prop(scene, "mode", expand=True)

        # Resolution Selector
        row = layout.row()
        row.prop(scene, "resolution", expand=True)

        # Start tracking button
        layout.label(text="Start Tracking:")
        row = layout.row()
        row.operator("point_tracking.start_tracking", text="Go !")


class SelectPoints(Operator):
    bl_idname = "point_tracking.select_points"
    bl_label = "Select Points"

    def execute(self, context):
        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected")
            return {"CANCELLED"}

        # Get selected tracks
        selected_tracks = [track for track in clip.tracking.tracks if track.select]

        # Clear existing selected tracks
        context.scene.selected_tracks.clear()

        # Add selected track names to the collection
        for track in selected_tracks:
            item = context.scene.selected_tracks.add()
            item.name = track.name

        # Update the display
        num_selected = len(selected_tracks)
        context.scene.selected_points_display = (
            f"{num_selected} point{'s' if num_selected != 1 else ''} selected"
        )

        self.report(
            {"INFO"},
            f"{num_selected} point{'s' if num_selected != 1 else ''} selected.",
        )

        return {"FINISHED"}


class SelectIsolatedPoints(Operator):
    bl_idname = "point_tracking.select_isolated_points"
    bl_label = "Select Isolated Points"
    bl_description = (
        "Select tracking points that have only one marker (isolated points)"
    )

    def execute(self, context):
        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected")
            return {"CANCELLED"}

        # Find isolated tracks (tracks with only one marker)
        isolated_tracks = [
            track for track in clip.tracking.tracks if len(track.markers) == 1
        ]

        # Clear existing selection
        for track in clip.tracking.tracks:
            track.select = False

        # Select isolated tracks
        for track in isolated_tracks:
            track.select = True

        # Clear existing selected_tracks in the scene
        context.scene.selected_tracks.clear()

        # Add selected isolated track names to the collection
        for track in isolated_tracks:
            item = context.scene.selected_tracks.add()
            item.name = track.name

        # Update the display
        num_selected = len(isolated_tracks)
        context.scene.selected_points_display = (
            f"{num_selected} isolated point{'s' if num_selected != 1 else ''} selected"
        )

        self.report(
            {"INFO"},
            f"{num_selected} isolated point{'s' if num_selected != 1 else ''} selected.",
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
        self.total_frames = 0  # Will store the total frames to process

    def _build_model(self, context):
        selected_device = context.scene.device
        if selected_device == "GPU":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
                print("GPU not available, defaulting to CPU mode.")
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
        selected_track_names = [item.name for item in context.scene.selected_tracks]
        if not selected_track_names:
            self.report({"ERROR"}, "No points selected. Please select points first.")
            return None

        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected")
            return None

        tracks = clip.tracking.tracks
        selected_tracks = [
            track for track in tracks if track.name in selected_track_names
        ]

        if not selected_tracks:
            self.report({"ERROR"}, "Selected tracks not found in the current clip.")
            return None

        return selected_tracks

    def _extract_query_points(self, tracks):
        try:
            first_markers = [track.markers[0] for track in tracks]
        except IndexError:
            self.report({"ERROR"}, "One or more tracks do not have markers.")
            return None

        frames = torch.tensor(
            [marker.frame for marker in first_markers], dtype=torch.int32
        )
        co_tensor = torch.stack(
            [torch.tensor(marker.co, dtype=torch.float32) for marker in first_markers],
            dim=0,
        )
        query_points = torch.cat((frames.unsqueeze(1), co_tensor), dim=1)
        return query_points

    def _get_video_frames(self, clip):
        video_path = bpy.path.abspath(clip.filepath)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            self.report({"ERROR"}, f"Unable to open video file: {video_path}")
            return None

        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                self.report({"ERROR"}, f"Failed to read frame {i} from video.")
                break
            frames.append(frame)

        cap.release()
        return frames

    @torch.no_grad()
    def _online_tracking_init(self, context):
        query_points = self._extract_query_points(self.current_tracks)
        if query_points is None:
            return False

        self.start_frame = int(query_points[:, 0].min().item())
        self.frames = self.frames[self.start_frame - 1 :]
        query_points[:, 0] -= self.start_frame
        self.tracker.init_tracker(self.frames, query_points)
        return True

    @torch.no_grad()
    def _online_tracking_step(self, context):
        # Check if all frames have been processed
        if self.frame_index >= len(self.frames):
            return False

        # Report progress in the console or Info bar
        current_frame_global = self.frame_index + self.start_frame
        self.report(
            {"INFO"},
            f"Tracking frame {current_frame_global}/{self.start_frame + self.total_frames - 1}",
        )

        frame = self.frames[self.frame_index]
        pred_tracks, pred_visibility = self.tracker.predict_frame([frame])

        for i in range(len(pred_tracks)):
            if pred_visibility[i]:
                coord = pred_tracks[i].cpu().numpy()
                coord[1] = 1 - coord[1]  # Flip Y to match Blender coords

                if 0 <= coord[0] <= 1 and 0 <= coord[1] <= 1:
                    self.current_tracks[i].markers.insert_frame(
                        current_frame_global
                    ).co = coord
                else:
                    print("Some coordinates are outside of the frame boundaries.")

        # Update the displayed frame in the movie-clip editor and timeline
        context.scene.frame_current = current_frame_global
        for area in context.screen.areas:
            if area.type == "CLIP_EDITOR":
                for space in area.spaces:
                    if space.type == "CLIP_EDITOR" and space.clip_user is not None:
                        space.clip_user.frame_current = current_frame_global

        # Update Blender's progress bar
        context.window_manager.progress_update(current_frame_global)

        self.frame_index += 1
        return True

    @torch.no_grad()
    def _offline_tracking(self, frames, current_tracks):
        raise NotImplementedError

    def invoke(self, context, event):
        if not self._build_model(context):
            return {"CANCELLED"}

        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected")
            return {"CANCELLED"}

        self.frames = self._get_video_frames(clip)
        if self.frames is None:
            return {"CANCELLED"}

        # Keep track of total frames for progress bar
        self.total_frames = len(self.frames)

        self.current_tracks = self._retrieve_selected_tracks(context)
        if self.current_tracks is None:
            return {"CANCELLED"}

        if context.scene.mode == "Online":
            if not self._online_tracking_init(context):
                return {"CANCELLED"}
        else:
            raise NotImplementedError("Offline tracking not yet implemented")

        # Start the progress bar: min=0, max=total_frames
        print(self.start_frame, self.total_frames)
        context.window_manager.progress_begin(self.start_frame, self.total_frames)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            self.report({"WARNING"}, "Tracking canceled by user.")
            context.window_manager.progress_end()  # End the progress bar on cancel
            return {"CANCELLED"}

        if context.scene.mode == "Online":
            running = self._online_tracking_step(context)
            if not running:
                self.report({"INFO"}, "Tracking completed successfully.")
                # End the progress bar once finished
                context.window_manager.progress_end()
                return {"FINISHED"}
        else:
            raise NotImplementedError("Offline modal stepping not yet implemented")

        return {"RUNNING_MODAL"}


# Registration
classes = (
    SelectedTrackItem,
    MotionTrackingPanel,
    SelectPoints,
    SelectIsolatedPoints,
    StartTracking,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    init_properties()


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    clear_properties()
