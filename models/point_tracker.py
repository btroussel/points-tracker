import os
from abc import ABC, abstractmethod
from PIL import Image
import cv2
from torchvision import transforms

import torch
import torch.nn.functional as F

from .tapnet import tapir_model


class BasePointTracker(ABC):
    @abstractmethod
    def __init__(self, model_type="tapir"):
        self.base_path = os.path.dirname(__file__)
        self.checkpoints_path = os.path.join(self.base_path, "resources", "checkpoints")

    def model_predict(self, frames, query_points):
        pass


class TAPIR(BasePointTracker):
    def __init__(self):
        super().__init__()

        self.model_path = os.path.join(
            self.checkpoints_path, "causal_bootstapir_checkpoint.pt"
        )
        self.use_casual_conv = True

        # Transformations
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Lambda(lambda x: x / 255 * 2 - 1),
            ]
        )

        # Load model
        self.model = tapir_model.TAPIR(
            pyramid_level=1, use_casual_conv=self.use_casual_conv
        )
        self.model.load_state_dict(torch.load(self.model_path))

    def _preprocess_frame(self, frame):
        return self.preprocess(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0)

    def _postprocess_occlusions(occlusions, expected_dist):
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
        return visibles

    def predict_frame(self, frame, query_points):
        """Compute point tracks and occlusions given frames and query points."""
        frame = self._preprocess_frame(frame).unsqueeze(0).to(self.model.device)
        print(frame)
        feature_grids = self.model.get_feature_grids(frame, is_training=False)
        query_features = self.model.get_query_features(
            frame,
            is_training=False,
            query_points=query_points,
            feature_grids=feature_grids,
        )
        if not hasattr(self, "causal_context"):
            self.causal_context = self.model.construct_initial_causal_state(
                query_points.shape[0], len(query_features.resolutions) - 1
            )

        trajectories = self.model.estimate_trajectories(
            frame.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=query_features,
            query_points_in_video=None,
            query_chunk_size=64,
            causal_context=self.causal_context,
            get_causal_context=True,
        )
        self.causal_context = trajectories["causal_context"]
        del trajectories["causal_context"]
        # Take only the predictions for the final resolution.
        # For running on higher resolution, it's typically better to average across
        # resolutions.
        tracks = trajectories["tracks"][-1]
        occlusions = trajectories["occlusion"][-1]
        uncertainty = trajectories["expected_dist"][-1]
        visibles = self._postprocess_occlusions(occlusions, uncertainty)
        return tracks, visibles
