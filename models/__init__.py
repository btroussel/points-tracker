from .point_tracker import TAPIR


def build_point_tracker(tracker_type):
    if tracker_type == "tapir":
        return TAPIR()
    else:
        raise NotImplementedError(f"Model {tracker_type} not implemented")
