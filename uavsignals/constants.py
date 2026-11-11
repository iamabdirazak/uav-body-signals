"""Signal vocabulary, skeleton layout and timing shared by every part of the pipeline."""

#: The 13 NATOPS aircraft-handling signals in UAV-GESTURE, in label order.
CLASSES = [
    'all_clear', 'have_command', 'hover', 'land', 'landing_direction', 'move_ahead',
    'move_downward', 'move_to_left', 'move_to_right', 'move_upward', 'not_clear',
    'slow_down', 'wave_off',
]
NUM_CLASSES = len(CLASSES)
HAVE_COMMAND = CLASSES.index('have_command')   # attention signal: arms the safety gate
WAVE_OFF = CLASSES.index('wave_off')           # abort: always executed, never gated

#: 14 body joints, COCO-18 order with eyes and ears dropped:
#: nose, neck, R shoulder, R elbow, R wrist, L shoulder, L elbow, L wrist,
#: R hip, R knee, R ankle, L hip, L knee, L ankle.
JOINT_NAMES = [
    'nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist',
    'r_hip', 'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle',
]
NUM_JOINTS = len(JOINT_NAMES)
NECK, R_HIP, L_HIP = 1, 8, 11
R_ARM, L_ARM = (2, 3, 4), (5, 6, 7)

#: Bones of the 14-joint skeleton (used by the graph model and for drawing).
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13)]

FPS = 25                 # keypoint rate the model expects (Hz)
WINDOW = 30              # frames per classification window (1.2 s)
FEATURE_DIM = NUM_JOINTS * 2 * 2   # position + velocity per joint = 56
