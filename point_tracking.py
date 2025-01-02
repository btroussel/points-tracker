import bpy
import os
import torch
import cv2
from PIL import Image
from torchvision import models, transforms
from bpy.types import Panel, Operator, PropertyGroup

from .models import build_model
from .utils import get_vars_from_context


# Define a PropertyGroup to store track names
class SelectedTrackItem(PropertyGroup):
    name: bpy.props.StringProperty(name="Track Name")


def init_properties():
    bpy.types.Scene.selected_points_display = bpy.props.StringProperty(
        name="Selected Points Display",
        description="Display the number of selected points",
        default="",
    )

    bpy.types.Scene.selected_tracks = bpy.props.CollectionProperty(
        type=SelectedTrackItem
    )

    bpy.types.Scene.bool_setting = bpy.props.BoolProperty(
        name="Enable Feature", description="A boolean setting example", default=False
    )

    bpy.types.Scene.enum_setting = bpy.props.EnumProperty(
        name="Mode",
        description="Choose a mode",
        items=[
            ("OPTION_A", "Option A", ""),
            ("OPTION_B", "Option B", ""),
            ("OPTION_C", "Option C", ""),
        ],
        default="OPTION_A",
    )

    bpy.types.Scene.float_setting = bpy.props.FloatProperty(
        name="Tracking Speed",
        description="Adjust the tracking speed",
        default=1.0,
        min=0.1,
        max=10.0,
    )

    bpy.types.Scene.int_setting = bpy.props.IntProperty(
        name="Tracking Resolution",
        description="Set the tracking resolution",
        default=1080,
        min=480,
        max=4320,
    )


def clear_properties():
    del bpy.types.Scene.selected_points_display
    del bpy.types.Scene.selected_tracks
    del bpy.types.Scene.bool_setting
    del bpy.types.Scene.enum_setting
    del bpy.types.Scene.float_setting
    del bpy.types.Scene.int_setting


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
        layout.label(text="Point Selection")
        row = layout.row()
        row.operator("point_tracking.select_points", text="Select Points")

        # Display selected points
        layout.label(text="Selected Points:")
        row = layout.row()
        row.prop(scene, "selected_points_display", text="")

        # Start tracking button
        row = layout.row()
        row.operator("point_tracking.start_tracking", text="Start Tracking")

        ## Settings
        layout.separator()
        layout.label(text="Settings")

        # Boolean setting (checkbox)
        row = layout.row()
        row.prop(scene, "bool_setting", text="Enable Feature")

        # Enum setting (dropdown)
        row = layout.row()
        row.prop(scene, "enum_setting", text="Mode")

        # Float setting (slider)
        row = layout.row()
        row.prop(scene, "float_setting", text="Tracking Speed")

        # Integer setting (number input)
        row = layout.row()
        row.prop(scene, "int_setting", text="Tracking Resolution")

        # TODO : Create spline for tracking to have more tracks
        # TODO : Integrate a loading bar for tracking


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

        # TODO : Only select isolated tracking points ? Assert

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


class StartTracking(Operator):
    bl_idname = "point_tracking.start_tracking"
    bl_label = "Start Tracking"
    bl_options = {"REGISTER", "UNDO"}

    def __init__(self):
        super().__init__()
        self.model_type = "tapir"

    def _build_model(self):
        self.model = build_model(self.model_type)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

        return True

    def _retrieve_selected_tracks(self, context):
        # TODO : I could use this function in other methods ?
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

        clip = context.edit_movieclip
        if clip is None:
            self.report({"ERROR"}, "No movie clip selected")
            return None

        # Retrieve all tracks from the clip
        tracks = clip.tracking.tracks

        # Find tracks that match the selected names
        selected_tracks = [
            track for track in tracks if track.name in selected_track_names
        ]

        if not selected_tracks:
            self.report({"ERROR"}, "Selected tracks not found in the current clip.")
            return None

        return selected_tracks

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

    def _update_tracks(self, tracks, model_output, frame_number):
        """
        Updates the tracks with the model output.

        Args:
            tracks (List[bpy.types.MovieTrackingTrack]): The tracks to update.
            model_output: The output from the AI model.
            frame_number (int): The current frame number.
        """
        # Interpret model_output and update tracks accordingly
        # This will depend on your model's output format

        # Example: Suppose model_output contains new positions for each track
        for track in tracks:
            # Get the new position for the track
            # This is just a placeholder example
            new_position = (0.5, 0.5)  # Replace with actual data from model_output

            # Insert the marker at the current frame
            marker = track.markers.insert(frame_number)
            marker.co = new_position  # Set the new position

            # Optionally set additional marker properties

    def execute(self, context):
        # Load the AI model
        if not self._build_model():
            return {"CANCELLED"}

        # Retrieve selected tracks
        selected_tracks = self._retrieve_selected_tracks(context)
        if selected_tracks is None:
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

        # Process frames with the AI model
        for frame_number, frame in enumerate(frames):

            output = self.model.predict_frame(frame)  # tracks, visibles
            print(output)
            self._update_tracks(selected_tracks, output, frame_number)

        self.report({"INFO"}, "Tracking completed successfully.")
        return {"FINISHED"}


# Registration
classes = (
    SelectedTrackItem,
    MotionTrackingPanel,
    SelectPoints,
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
