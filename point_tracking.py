import bpy
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F
from torchvision import models, transforms
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
            ("256x256", "256x256", "Set resolution to 256x256 pixels"),
            ("512x512", "512x512", "Set resolution to 512x512 pixels"),
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
        row = layout.row()
        row.prop(scene, "mode", expand=True)

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
            return True
        except Exception as e:
            self.report({"ERROR"}, f"Failed to build tracker: {e}")
            return False

    def _retrieve_selected_tracks(self, context):
        """
        Retrieves the selected tracks from the scene's stored selection.

        Returns:
            List[bpy.types.MovieTrackingTrack]: A list of selected tracks.
            Returns None if no tracks are selected or if there's an error.
        """
        # Get the selected track names from the scene property
        selected_track_names = [item.name for item in context.scene.selected_tracks]
        if not selected_track_names:
            self.report({"ERROR"}, "No points selected. Please select points first.")
            return None

        # Retrieve all track from the clip that match names
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
        """
        Args : tracks (List[bpy.types.MovieTrackingTrack]): The tracks to extract query points from.
        Format of markers : [W, H]

        Returns: query_points (torch.Tensor): The query points as a tensor of shape {1 N (T H W)}

        """

        try:
            first_markers = [track.markers[0] for track in tracks]
        except IndexError:
            self.report({"ERROR"}, "One or more tracks do not have markers.")
            return None

        frames = torch.tensor(
            [marker.frame for marker in first_markers],
            dtype=torch.int32,
        )
        co_tensor = torch.stack(
            [torch.tensor(marker.co, dtype=torch.float32) for marker in first_markers],
            dim=0,
        )
        query_points = torch.cat((frames.unsqueeze(1), co_tensor), dim=1)

        return query_points

    def _get_video_frames(self, clip):
        """
        Retrieves all frames from the video clip using OpenCV.

        Args:
            clip (bpy.types.MovieClip): The movie clip to read frames from.

        Returns:
            List[np.ndarray]: A list of frames as numpy arrays.
        """
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
    def _online_tracking(self, frames, current_tracks, context):

        # Extract query points
        query_points = self._extract_query_points(current_tracks)
        start_frame = int(query_points[:, 0].min().item())
        frames = frames[start_frame - 1 :]
        query_points[:, 0] -= start_frame

        self.tracker.init_tracker(frames, query_points)

        for frame_number, frame in tqdm(enumerate(frames)):
            pred_tracks, pred_visibility = self.tracker.predict_frame([frame])
            for i in range(len(pred_tracks)):
                if pred_visibility[i]:
                    coord = pred_tracks[i].cpu().numpy()
                    coord[1] = 1 - coord[1]

                    # Assuming coord has exactly two elements
                    if 0 <= coord[0] <= 1 and 0 <= coord[1] <= 1:
                        current_tracks[i].markers.insert_frame(
                            frame_number + start_frame
                        ).co = coord
                    else:
                        print("Some values have been outside of the frames.")
        return

    @torch.no_grad()
    def _offline_tracking(self, frames, current_tracks):
        raise NotImplementedError
        return

    def execute(self, context):
        # Load the AI model
        if not self._build_model(context):
            return {"CANCELLED"}

        # Get the movie clip
        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected")
            return {"CANCELLED"}

        # Get video frames
        frames = self._get_video_frames(clip)
        if frames is None:
            return {"CANCELLED"}

        # Retrieve selected tracks
        current_tracks = self._retrieve_selected_tracks(context)
        if current_tracks is None:
            return {"CANCELLED"}

        if context.scene.mode == "Online":
            self._online_tracking(frames, current_tracks, context)
        else:
            self._offline_tracking(frames, current_tracks)

        self.report({"INFO"}, "Tracking completed successfully.")
        return {"FINISHED"}


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
