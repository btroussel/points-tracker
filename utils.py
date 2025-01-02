def get_vars_from_context(context):
    scene = context.scene
    clip = context.area.spaces.active.clip
    if clip is None:
        return scene, None, None, None, None
    tracks = clip.tracking.tracks
    current_frame = scene.frame_current
    frame_duration = clip.frame_duration
    return scene, clip, tracks, current_frame, frame_duration
