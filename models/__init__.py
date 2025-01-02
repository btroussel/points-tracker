from .point_tracker import TAPIR


def build_model(model_type):
    if model_type == "tapir":
        return TAPIR()
    else:
        raise NotImplementedError(f"Model {model_type} not implemented")
