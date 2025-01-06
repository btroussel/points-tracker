from .point_tracker import TAPIR


def build_point_tracker(tracker_type, mode, resolution, device):
    if tracker_type == "tapir":
        return TAPIR(mode, resolution, device)
    else:
        raise NotImplementedError(f"Model {tracker_type} not implemented")
