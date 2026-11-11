"""Finding a camera that actually delivers a picture."""
import platform
import time


def open_capture(index):
    import cv2
    backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def is_black(frame):
    return frame is None or float(frame.mean()) < 4.0


def find_camera(preferred=-1):
    """Try the preferred index, then 0 to 4, skipping cameras that only give black frames."""
    candidates = [preferred] if preferred is not None and preferred >= 0 else []
    candidates += [i for i in range(5) if i not in candidates]
    for i in candidates:
        cap = open_capture(i)
        if not cap.isOpened():
            continue
        good = 0
        deadline = time.time() + 2.5        # cameras often send dark frames while starting
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and not is_black(frame):
                good += 1
                if good >= 3:
                    print(f'Using camera {i}.')
                    return cap
        cap.release()
    return None
