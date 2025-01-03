import os
from abc import ABC, abstractmethod
from PIL import Image
import cv2
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tapnet import tapir_model


class BasePointTracker(ABC):
    @abstractmethod
    def __init__(self, model_type="tapir"):
        self.checkpoints_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir,
            "resources",
            "checkpoints",
        )

    def init_tracker(self, frames, query_points, device):
        pass

    def model_predict(self, frame, device):
        pass


class TAPIR(BasePointTracker, nn.Module):
    def __init__(self):
        BasePointTracker.__init__(self)
        nn.Module.__init__(self)

        self.model_path = os.path.join(
            self.checkpoints_path, "causal_bootstapir_checkpoint.pt"
        )
        self.use_causal_conv = True

        # Transformations
        # TODO : Speed up with a dataloader for parallel processing
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
            pyramid_level=1, use_causal_conv=self.use_causal_conv
        )
        self.model.load_state_dict(torch.load(self.model_path))

    def _preprocess_frames(self, frames):
        tensors = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(self.preprocess(Image.fromarray(rgb_frame)))
        return torch.stack(tensors).permute(0, 2, 3, 1)

    def _postprocess_occlusions(self, occlusions, expected_dist):
        visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
        return visibles

    def init_tracker(self, frames, query_points, device):
        """
        frames list(np.array): List of frames {N (H W C)}
        query_points {1 N (T H W)}
        """

        # Process frames
        end_frame = query_points[0, :, 0].max().item()
        frames = frames[: end_frame + 1]

        frames = self._preprocess_frames(frames).unsqueeze(0).to(device)
        feature_grids = self.model.get_feature_grids(frames, is_training=False)
        self.query_features = self.model.get_query_features(
            frames,
            is_training=False,
            query_points=query_points,
            feature_grids=feature_grids,
        )
        self.causal_context = self.model.construct_initial_causal_state(
            query_points.shape[1], len(self.query_features.resolutions) - 1
        )

        for i in range(len(self.causal_context)):
            for k, v in self.causal_context[i].items():
                self.causal_context[i][k] = v.to(device)

    def predict_frame(self, frame, device):
        """Compute point tracks and occlusions given frames and query points."""
        # TODO : Use query points in video ?

        frame = self._preprocess_frames(frame).unsqueeze(0).to(device)
        feature_grids = self.model.get_feature_grids(frame, is_training=False)
        trajectories = self.model.estimate_trajectories(
            frame.shape[-3:-1],
            is_training=False,
            feature_grids=feature_grids,
            query_features=self.query_features,
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
        return tracks[0, :, 0], visibles[0, :, 0]


# https://colab.research.google.com/github/deepmind/tapnet/blob/master/colabs/torch_causal_tapir_demo.ipynb#scrollTo=_LLK7myqp3Px
