"""Live demo: command recognition from a webcam or a video file, with on-screen guidance.

    python demo/run.py                   # finds a working camera
    python demo/run.py --camera 1        # a specific camera
    python demo/run.py --video clip.mp4  # a recorded video

Step 1, perform `have command` until Output shows Ready. Step 2, perform the signal shown in Guide.
Keys: N / P next or previous signal, V switch view, R reset, Q or Esc quit.

The model was trained on OpenPose keypoints from drone video. Here MediaPipe supplies keypoints from
a ground-level camera, so accuracy differs from the paper. This is a demonstration, not an evaluation.
"""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from uavsignals import CLASSES, Recogniser  # noqa: E402
from uavsignals.constants import HAVE_COMMAND, WAVE_OFF  # noqa: E402
from uavsignals.recogniser import DEFAULT_WEIGHTS  # noqa: E402

from camera import find_camera, is_black  # noqa: E402
from pose import PoseEstimator  # noqa: E402
from ui import UI  # noqa: E402

PRACTICE = [c for c in range(len(CLASSES)) if c not in (HAVE_COMMAND, WAVE_OFF)]
KEYS = {'n': 'next', 'p': 'prev', 'r': 'reset', 'q': 'quit', 'v': 'view'}
WINDOW_TITLE = 'UAV Signal Recognition'


def parse_args():
    ap = argparse.ArgumentParser(description='Live body-signal recognition demo')
    ap.add_argument('--camera', type=int, default=-1, help='camera index; -1 finds one automatically')
    ap.add_argument('--video', default=None, help='a video file instead of a camera')
    ap.add_argument('--model', default=DEFAULT_WEIGHTS)
    ap.add_argument('--pose-model', default=os.path.join(HERE, 'pose_landmarker_lite.task'))
    ap.add_argument('--theme', choices=['auto', 'light', 'dark'], default='auto')
    return ap.parse_args()


def open_source(args):
    import cv2
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = find_camera(args.camera)
        if cap is None:
            sys.exit('No camera gave a picture. On macOS: System Settings > Privacy & Security > Camera, allow '
                     'your terminal app, then quit and reopen it. Close other apps using the camera, or try --camera 1.')
    if not cap.isOpened():
        sys.exit('Could not open the video.')
    return cap


def main():
    import cv2
    args = parse_args()
    pose = PoseEstimator(args.pose_model)
    recogniser = Recogniser.from_pretrained(args.model)
    with open(os.path.join(HERE, 'reference_signals.json')) as f:
        ui = UI(json.load(f), args.theme)
    cap = open_source(args)

    state = dict(armed=False, target=CLASSES.index('hover'), log=collections.deque(maxlen=20),
                 probs=None, camera_msg=('', True))
    clicks = []

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | getattr(cv2, 'WINDOW_FREERATIO', 0))
    cv2.resizeWindow(WINDOW_TITLE, 1440, 810)

    def on_mouse(event, x, y, flags, param):
        ui.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            action = ui.click(x, y)
            if action:
                clicks.append(action)
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse)

    start, black_frames, size, shown = time.time(), 0, (1440, 810), False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if args.video else time.time() - start
        black_frames = black_frames + 1 if is_black(frame) else 0
        keypoints = visible = None
        if black_frames > 30:
            state['camera_msg'] = ('Camera shows black: check camera permission, or try --camera 1', False)
        else:
            keypoints, visible = pose(frame, t)
            if keypoints is None:
                state['camera_msg'] = ('No person detected: step into view', False)
            else:
                legs_seen = visible[[9, 10, 12, 13]].sum() >= 3
                state['camera_msg'] = ('' if legs_seen else 'Step back: show your whole body, head to ankles', legs_seen)
                out = recogniser.push(t, keypoints, visible)
                if out is not None:
                    state['probs'], state['armed'] = out.probs, out.armed
                    if out.command:
                        state['log'].append(out.command)
                        if out.command == CLASSES[state['target']]:   # practised signal done: next one
                            state['target'] = PRACTICE[(PRACTICE.index(state['target']) + 1) % len(PRACTICE)]

        try:   # draw at the window's current size, so resizing never stretches the picture
            rect = cv2.getWindowImageRect(WINDOW_TITLE)
            if rect[2] > 200 and rect[3] > 200:
                size = (rect[2], rect[3])
        except Exception:
            pass
        cv2.imshow(WINDOW_TITLE, ui.render(frame, keypoints, visible, state, size))

        key = cv2.waitKey(1) & 0xFF
        action = clicks.pop(0) if clicks else KEYS.get(chr(key).lower() if key < 128 else '')
        try:   # the window's close button ends the demo
            visible_flag = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE)
            shown = shown or visible_flag >= 1
            if shown and visible_flag < 1:
                break
        except Exception:
            pass
        if action == 'quit' or key == 27:
            break
        if action == 'reset':
            recogniser.reset()
            state.update(armed=False, probs=None)
        if action == 'view':
            ui.toggle_view()
        if action in ('next', 'prev') and state['armed']:
            i = PRACTICE.index(state['target']) if state['target'] in PRACTICE else 0
            state['target'] = PRACTICE[(i + (1 if action == 'next' else -1)) % len(PRACTICE)]
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
