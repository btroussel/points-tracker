from .point_tracker import TAPIR


def build_point_tracker(tracker_type, device):
    if tracker_type == "tapir":
        return TAPIR(device)
    else:
        raise NotImplementedError(f"Model {tracker_type} not implemented")
