"""A test-time viewpoint model that training never sees: lift the skeleton to 3D, then view it from a drone.

Training simulates elevation by squashing the body vertically (augment.elevation). To test viewpoint
robustness independently of that model, each recorded skeleton is instead

  1. lifted to 3D. The torso (neck, shoulders, hips) is kept in the image plane. With `limb_depth`, every
     limb segment also gets the depth that makes its 3D length equal to its longest 2D length in the
     clip, the usual bone-length lifting for a person who faces the camera. Arms and the head are then
     placed toward the camera, the natural choice for a person signalling to a drone in front of them.
     Without it, the whole body stays flat in the image plane, so only perspective changes the view.
  2. placed standing on the ground, scaled to metres with an assumed torso length of 0.52 m.
  3. projected through a pinhole camera at a given range and elevation angle, aimed at the person's
     mid-body. This captures perspective and depth, which the training augmentation ignores.
"""
import numpy as np

from .constants import L_HIP, NECK, R_HIP
from .features import fill_missing, torso_length

TORSO_M = 0.52            # neck to mid-hip, adult average (metres)
FOCAL_PX = 1000.0         # pinhole focal length for a 1920 x 1080 image
CENTRE = np.array([960.0, 540.0])
AIM_HEIGHT_M = 1.0        # the camera looks at this height on the person

# (parent, child, depth sign): +1 places the child toward the camera
SEGMENTS = [(NECK, 0, 1), (2, 3, 1), (3, 4, 1), (5, 6, 1), (6, 7, 1),
            (R_HIP, 9, 0), (9, 10, 0), (L_HIP, 12, 0), (12, 13, 0)]


def lift(x, visible, limb_depth=True):
    """(T, 14, 2) pixel keypoints -> (T, 14, 3) metres. X right, Y up, Z toward the original camera."""
    x = fill_missing(x, visible)
    metres = TORSO_M / (torso_length(x) + 1e-6)
    P = np.zeros(x.shape[:2] + (3,), np.float32)
    P[..., 0] = (x[..., 0] - x[:, NECK:NECK + 1, 0]) * metres
    P[..., 1] = -(x[..., 1] - x[:, NECK:NECK + 1, 1]) * metres
    for parent, child, sign in (SEGMENTS if limb_depth else []):
        planar = np.linalg.norm(P[:, child, :2] - P[:, parent, :2], axis=-1)
        length = planar.max()
        P[:, child, 2] = P[:, parent, 2] + sign * np.sqrt(np.maximum(length ** 2 - planar ** 2, 0))
    P[..., 1] -= P[..., 1][:, [10, 13]].min()          # feet on the ground
    return P


def camera_position(range_m, elevation_deg):
    """Camera `range_m` from the aim point, raised by `elevation_deg` above the horizontal."""
    theta = np.deg2rad(elevation_deg)
    return np.array([0.0, AIM_HEIGHT_M + range_m * np.sin(theta), range_m * np.cos(theta)])


def project(P, range_m, elevation_deg):
    """View the 3D skeleton from a camera `range_m` away, looking down at `elevation_deg`."""
    cam = camera_position(range_m, elevation_deg)
    forward = np.array([0.0, AIM_HEIGHT_M, 0.0]) - cam
    forward /= np.linalg.norm(forward)
    right = np.array([1.0, 0.0, 0.0])                    # same side as the original camera, so image x keeps its direction
    up = np.cross(right, forward)
    rel = P - cam
    depth = rel @ forward
    u = CENTRE[0] + FOCAL_PX * (rel @ right) / depth
    v = CENTRE[1] - FOCAL_PX * (rel @ up) / depth
    return np.stack([u, v], -1).astype(np.float32)


def drone_view(range_m, elevation_deg, limb_depth=True):
    """Return f(x, visible, rng) that re-views a recorded clip from the given camera position.
    Joints missing in the recording are interpolated before lifting, so every joint is visible afterwards."""
    def view(x, visible, rng=None):
        return project(lift(x, visible, limb_depth), range_m, elevation_deg), np.ones(visible.shape, bool)
    return view
