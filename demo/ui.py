"""Window drawing for the live demo: Guide (left), camera or pose canvas (centre), Output (right).

Drawn with Pillow at whatever size the window has, in light or dark to follow the system setting.
"""
import collections
import os
import platform
import time

import numpy as np

from uavsignals.constants import CLASSES, EDGES, HAVE_COMMAND

PRETTY = {c: c.replace('_', ' ').capitalize() for c in CLASSES}

ACC = (62, 155, 200)          # one accent in both themes
TINTS = {   # (fill colour, text colour) for tinted secondary buttons, per theme
    'light': {'accent': (ACC, (35, 115, 159)), 'orange': ((255, 149, 0), (194, 104, 0)), 'red': ((255, 59, 48), (204, 40, 32))},
    'dark': {'accent': (ACC, (122, 192, 228)), 'orange': ((255, 159, 10), (255, 179, 64)), 'red': ((255, 69, 58), (255, 105, 97))},
}
THEMES = {
    'light': dict(bg=(242, 242, 247), card=(255, 255, 255), ink=(29, 29, 31), ink2=(110, 110, 115), ink3=(174, 174, 178),
                  track=(229, 229, 234), seg=(227, 227, 232), trail=(181, 218, 236), on_acc=(226, 241, 250), canvas=(252, 252, 253), dot=(214, 214, 220)),
    'dark': dict(bg=(22, 22, 24), card=(44, 44, 46), ink=(245, 245, 247), ink2=(152, 152, 157), ink3=(99, 99, 102),
                 track=(62, 62, 66), seg=(44, 44, 46), trail=(40, 96, 124), on_acc=(226, 241, 250), canvas=(30, 30, 32), dot=(62, 62, 66)),
}


def system_theme():
    """'dark' or 'light', following the operating system setting."""
    try:
        if platform.system() == 'Darwin':
            import subprocess
            r = subprocess.run(['defaults', 'read', '-g', 'AppleInterfaceStyle'], capture_output=True, text=True, timeout=1)
            return 'dark' if 'Dark' in r.stdout else 'light'
        if platform.system() == 'Windows':
            import winreg
            k = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize')
            return 'light' if winreg.QueryValueEx(k, 'AppsUseLightTheme')[0] else 'dark'
    except Exception:
        pass
    return 'light'


class ThemeWatcher:
    """re-reads the system appearance every 2 s in the background, so switching light/dark applies live"""
    def __init__(self, mode):
        self.mode, self.value = mode, (mode if mode in THEMES else system_theme())
        if mode == 'auto':
            import threading
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            time.sleep(2); self.value = system_theme()


class Fonts:
    """system font (San Francisco on macOS) at a given weight and size"""
    CANDS = ['/System/Library/Fonts/SFNS.ttf', '/System/Library/Fonts/Helvetica.ttc', 'C:/Windows/Fonts/segoeui.ttf',
             '/Library/Fonts/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    WEIGHTS = {'r': 'Regular', 'm': 'Medium', 's': 'Semibold', 'b': 'Bold'}

    def __init__(self):
        self.path = next((p for p in self.CANDS if os.path.exists(p)), None); self.cache = {}

    def __call__(self, w, size):
        key = (w, size)
        if key not in self.cache:
            from PIL import ImageFont
            if self.path is None:
                self.cache[key] = ImageFont.load_default()
            else:
                f = ImageFont.truetype(self.path, size)
                try:
                    f.set_variation_by_name(self.WEIGHTS[w])
                except Exception:
                    pass
                self.cache[key] = f
        return self.cache[key]


class UI:
    """Draws the whole window at whatever size the window currently has.
    Side panels keep their proportions (scaled by u); the centre takes the remaining width."""
    VIEWS = ['Pose', 'Scene']

    def __init__(self, ref, theme_mode='auto'):
        self.F, self.ref, self.theme = Fonts(), ref, ThemeWatcher(theme_mode)
        self.buttons = {}              # clickable areas drawn this frame: name -> (x0, y0, x1, y1)
        self.mouse = (-1, -1)          # pointer position, for hover highlights
        self.view = 0                  # 0 = Pose (dotted canvas), 1 = Scene (camera image)
        self.view_pos = 0.0            # animated thumb of the view control
        self.trail = collections.deque(maxlen=40)   # wrist positions in camera pixels
        self.frame_box = None                       # smoothed Pose-view framing (cx, cy, scale)
        self.grid_cache = {}
        self.bounds = {c: (np.array(v).reshape(-1, 2).min(0), np.array(v).reshape(-1, 2).max(0)) for c, v in ref.items()}

    # ---------------------------------------------------------- helpers
    def toggle_view(self):
        self.view = 1 - self.view

    def click(self, x, y):
        """returns the name of the button that was clicked (the view control is handled here)"""
        for name, (x0, y0, x1, y1) in self.buttons.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                if name == 'view':
                    self.view = 0 if x < (x0 + x1) / 2 else 1
                    return None
                return name
        return None

    def _hover(self, r):
        return r[0] <= self.mouse[0] <= r[2] and r[1] <= self.mouse[1] <= r[3]

    def _button(self, d, name, rect, label, tint, base, radius, u, tname, enabled=True):
        """tinted secondary button: soft colour fill, coloured label; a little stronger on hover"""
        col, txt = TINTS[tname][tint]
        if enabled:
            self.buttons[name] = rect
        else:   # disabled: neutral fill, faint label, not clickable
            col, txt = THEMES[tname]['ink3'], THEMES[tname]['ink3']
        a = (0.14 if tname == 'light' else 0.22) + (0.08 if enabled and self._hover(rect) else 0.0)
        fill = tuple(int(round(base[i] * (1 - a) + col[i] * a)) for i in range(3))
        d.rounded_rectangle(rect, radius=radius, fill=fill)
        f = self.F('s', round(15 * u)); tw = d.textlength(label, font=f)
        d.text(((rect[0] + rect[2] - tw) / 2, (rect[1] + rect[3]) / 2 - f.size * 0.62), label, font=f, fill=txt)

    def _bar(self, d, x, y, w, v, track, fill, h=8):
        h = max(4, int(round(h)))
        d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=track)
        if v > 0.01:
            d.rounded_rectangle((x, y, x + max(h, int(w * v)), y + h), radius=h // 2, fill=fill)

    def _segmented(self, canvas, d, x, y, w, h, labels, pos, T, u, track, pad=4):
        from PIL import Image, ImageDraw
        F = self.F; fs = F('s', round(15 * u)); half = w / len(labels); r = int(16 * u)   # rounded rectangle; thumb radius = 16 - 4 = 12, same as the buttons
        d.rounded_rectangle((x, y, x + w, y + h), radius=r, fill=track)
        ty = y + h / 2 - fs.size * 0.62
        for i, lab in enumerate(labels):
            tw = d.textlength(lab, font=fs); d.text((x + i * half + (half - tw) / 2, ty), lab, font=fs, fill=T['ink2'])
        p = max(3, int(round(pad * u))); tx = int(round(x + p + pos * (half - p))); tw_, th_ = int(half - p), int(h - 2 * p)
        thumb = Image.new('RGB', (tw_, th_), ACC); td = ImageDraw.Draw(thumb)
        for i, lab in enumerate(labels):   # white label only where the thumb covers it
            lw = td.textlength(lab, font=fs); td.text((x + i * half + (half - lw) / 2 - tx, ty - y - p), lab, font=fs, fill=(255, 255, 255))
        mask = Image.new('L', (tw_, th_), 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, tw_ - 1, th_ - 1), radius=max(2, r - p), fill=255)   # concentric
        canvas.paste(thumb, (tx, int(y + p)), mask)

    def _figure(self, d, cls, cx, top, bot, T, u):
        seq = self.ref[cls]; lo, hi = self.bounds[cls]
        sc = min((bot - top) / max(hi[1] - lo[1], 1e-6), 230 * u / max(hi[0] - lo[0], 1e-6))
        X = lambda q: (cx - (q[0] - (lo[0] + hi[0]) / 2) * sc, bot - (q[1] - lo[1]) * sc)   # mirrored, like a mirror
        r1, lw, r2 = 3 * u, max(2, int(5 * u)), 5 * u
        for wr in (4, 7):
            for i in range(0, len(seq), 2):
                x, y = X(seq[i][wr]); d.ellipse((x - r1, y - r1, x + r1, y + r1), fill=T['trail'])
        P = [X(q) for q in seq[int(time.time() * 25) % len(seq)]]
        for a, b in EDGES:
            d.line([P[a], P[b]], fill=T['ink'], width=lw, joint='curve')
        for x, y in P:
            d.ellipse((x - r2, y - r2, x + r2, y + r2), fill=T['ink'])

    def _grid(self, cw, H, T, u, theme):
        key = (cw, H, theme, round(u, 2))
        if key not in self.grid_cache:
            self.grid_cache.clear()
            g = np.empty((H, cw, 3), np.uint8); g[:] = T['canvas'][::-1]
            step = max(14, int(28 * u)); r = max(1, int(round(1.6 * u)))
            for yy in range(step // 2, H, step):
                for xx in range((cw % step) // 2 + step // 2, cw, step):
                    cv2_circle(g, (xx, yy), r, T['dot'][::-1])
            self.grid_cache[key] = g
        return self.grid_cache[key].copy()

    # ---------------------------------------------------------- frame
    def render(self, frame, kp, m, st, size=None):
        import cv2
        from PIL import Image, ImageDraw
        F, tname = self.F, self.theme.value; T = THEMES[tname]
        self.buttons = {}
        h0, w0 = frame.shape[:2]
        W, H = size if size else (1600, 900)
        u = float(np.clip(min(H / 720, W / 1500), 0.8, 1.8))
        LEFT, RIGHT = int(340 * u), int(360 * u); cw = max(160, W - LEFT - RIGHT); W = LEFT + cw + RIGHT
        S = lambda v: int(round(v * u))

        # ---- centre: cover-fit the camera image (mirrored), or a dotted canvas
        sc = max(cw / w0, H / h0); ox, oy = (cw - w0 * sc) / 2, (H - h0 * sc) / 2
        to_c = lambda p: (int(round(ox + (w0 - p[0]) * sc)), int(round(oy + p[1] * sc)))
        if self.view == 1:
            img = cv2.resize(cv2.flip(frame, 1), (int(round(w0 * sc)), int(round(h0 * sc))), interpolation=cv2.INTER_AREA)
            x0, y0 = int(round(-ox)), int(round(-oy)); centre = img[y0:y0 + H, x0:x0 + cw].copy()
            if centre.shape[:2] != (H, cw):
                centre = cv2.resize(centre, (cw, H))
        else:
            centre = self._grid(cw, H, T, u, tname)
        if kp is not None:
            for j in (4, 7):
                if m[j]:
                    self.trail.append(tuple(kp[j]))
            if self.view == 0:
                # Pose view frames the body: centred, about 70% of the canvas height, smoothed over time
                vis = kp[m] if m.any() else kp
                (bx0, by0), (bx1, by1) = vis.min(0), vis.max(0)
                bh = max(by1 - by0, (bx1 - bx0) * 0.6, 40.0)
                tgt = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2, min(0.7 * H / bh, 0.8 * cw / max(bx1 - bx0, 1.0))])
                self.frame_box = tgt if self.frame_box is None else self.frame_box + (tgt - self.frame_box) * 0.15
                fcx, fcy, fsc = self.frame_box
                to_c = lambda p: (int(round(cw / 2 - (p[0] - fcx) * fsc)), int(round(H / 2 + (p[1] - fcy) * fsc)))
            P = [to_c(p) for p in kp]
            if self.view == 0:
                for q in self.trail:
                    cv2.circle(centre, to_c(q), S(4), T['trail'][::-1], -1, cv2.LINE_AA)
            for a, b in EDGES:
                if m[a] and m[b]:
                    cv2.line(centre, P[a], P[b], ACC[::-1], S(6 if self.view == 0 else 4), cv2.LINE_AA)
            for j in range(14):
                if m[j]:
                    if self.view == 0:
                        cv2.circle(centre, P[j], S(7), ACC[::-1], -1, cv2.LINE_AA)
                        cv2.circle(centre, P[j], S(4), (255, 255, 255), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(centre, P[j], S(5), (255, 255, 255), -1, cv2.LINE_AA)
        else:
            self.trail.clear(); self.frame_box = None

        canvas = Image.new('RGB', (W, H), T['bg'])
        canvas.paste(Image.fromarray(cv2.cvtColor(centre, cv2.COLOR_BGR2RGB)), (LEFT, 0))
        d = ImageDraw.Draw(canvas)

        # view control, top centre (Pose | Scene)
        self.view_pos += (self.view - self.view_pos) * 0.25
        vw, vh = S(240), S(40); vx, vy = LEFT + (cw - vw) / 2, S(18)
        self.buttons['view'] = (vx, vy, vx + vw, vy + vh)
        self._segmented(canvas, d, vx, vy, vw, vh, self.VIEWS, self.view_pos, T, u, T['card'])

        # message only when something needs fixing
        msg, good = st['camera_msg']
        if msg and not good:
            fm = F('m', S(17)); tw = d.textlength(msg, font=fm) + S(44); x0 = LEFT + (cw - tw) / 2
            d.rounded_rectangle((x0, H - S(72), x0 + tw, H - S(26)), radius=S(23), fill=T['card'])
            d.text((x0 + S(22), H - S(61)), msg, font=fm, fill=T['ink'])

        ready = st['armed']
        target = CLASSES[st['target']]
        showing = target if ready else CLASSES[HAVE_COMMAND]

        # ============ left panel: Guide
        x, pw = S(24), LEFT - S(48)
        d.text((x, S(22)), 'Guide', font=F('b', S(28)), fill=T['ink'])
        d.text((x, S(60)), 'What to do in front of the camera', font=F('r', S(15)), fill=T['ink2'])
        # step status (not a control: the model moves you to step 2 when it recognises Have command)
        sy0, sy1 = S(96), S(170)
        if ready:   # step 2: soft accent tint, so progress is visible at a glance
            d.rounded_rectangle((x, sy0, x + pw, sy1), radius=S(22), fill=tint(ACC, T['card'], 0.14 if tname == 'light' else 0.22))
        else:
            d.rounded_rectangle((x, sy0, x + pw, sy1), radius=S(22), fill=T['card'])
        d.text((x + S(20), sy0 + S(14)), f"Step {2 if ready else 1} of 2", font=F('m', S(14)), fill=T['ink2'])
        d.text((x + S(20), sy0 + S(34)), 'Give a command' if ready else 'Get attention', font=F('b', S(22)), fill=T['ink'])
        segw, segh, sg = S(22), S(6), S(5)   # two-step progress, top right
        for i in range(2):
            sx = x + pw - S(20) - (2 - i) * segw - (1 - i) * sg
            on = i == 0 or ready
            d.rounded_rectangle((sx, sy0 + S(20), sx + segw, sy0 + S(20) + segh), radius=segh // 2,
                                fill=ACC if on else T['track'])
        # current signal, animated like a mirror
        cy, cb = S(184), S(500)
        d.rounded_rectangle((x, cy, x + pw, cb), radius=S(22), fill=T['card'])
        d.text((x + S(20), cy + S(16)), 'Perform this signal', font=F('m', S(14)), fill=T['ink2'])
        d.text((x + S(20), cy + S(38)), PRETTY[showing], font=F('b', S(22)), fill=T['ink'])
        self._figure(d, showing, x + pw / 2, cy + S(90), cb - S(20), T, u)
        if ready:   # N picks another command (only in step 2; in step 1 Have command comes first)
            d.text((x + S(4), cb + S(14)), 'Press N for the next command', font=F('r', S(15)), fill=T['ink2'])
        # app actions, bottom of the panel
        gy, bh, gap = H - S(24) - S(40), S(40), S(10); half = (pw - gap) / 2
        self._button(d, 'reset', (x, gy, x + half, gy + bh), 'Reset', 'orange', T['bg'], S(12), u, tname)
        self._button(d, 'quit', (x + half + gap, gy, x + pw, gy + bh), 'Quit', 'red', T['bg'], S(12), u, tname)

        # ============ right panel: Output
        x, pw = LEFT + cw + S(24), RIGHT - S(48)
        d.text((x, S(22)), 'Output', font=F('b', S(28)), fill=T['ink'])
        d.text((x, S(60)), 'What the model recognises', font=F('r', S(15)), fill=T['ink2'])
        if ready:
            d.rounded_rectangle((x, S(96), x + pw, S(170)), radius=S(22), fill=ACC)
            d.ellipse((x + S(20), S(114), x + S(32), S(126)), fill=(255, 255, 255))
            d.text((x + S(42), S(104)), 'Ready', font=F('b', S(22)), fill=(255, 255, 255))
            d.text((x + S(20), S(138)), 'Accepting commands', font=F('r', S(15)), fill=T['on_acc'])
        else:
            d.rounded_rectangle((x, S(96), x + pw, S(170)), radius=S(22), fill=T['card'])
            d.ellipse((x + S(20), S(114), x + S(32), S(126)), fill=T['ink3'])
            d.text((x + S(42), S(104)), 'Standby', font=F('b', S(22)), fill=T['ink'])
            d.text((x + S(20), S(138)), 'Waiting for Have command', font=F('r', S(15)), fill=T['ink2'])
        d.rounded_rectangle((x, S(184), x + pw, S(408)), radius=S(22), fill=T['card'])
        d.text((x + S(20), S(200)), 'Recognised signal', font=F('m', S(14)), fill=T['ink2'])
        probs = st['probs']
        if probs is None:
            d.text((x + S(20), S(230)), 'Waiting for a full body…', font=F('r', S(17)), fill=T['ink3'])
        else:
            for r_, c in enumerate(np.argsort(-probs)[:3]):
                yy = S(228) + r_ * S(54); top = r_ == 0
                name = PRETTY[CLASSES[c]] + ('  ✓' if top and CLASSES[c] == showing else '')
                f17 = F('s' if top else 'r', S(17))
                d.text((x + S(20), yy), name, font=f17, fill=T['ink'] if top else T['ink2'])
                pct = f'{int(round(probs[c] * 100))}%'; fp = F('m', S(17)); tw = d.textlength(pct, font=fp)
                d.text((x + pw - S(20) - tw, yy), pct, font=fp, fill=T['ink'] if top else T['ink2'])
                self._bar(d, x + S(20), yy + S(28), pw - S(40), float(probs[c]), T['track'], ACC if top else T['ink3'], h=8 * u)
        rb = S(500)
        d.rounded_rectangle((x, S(422), x + pw, rb), radius=S(22), fill=T['card'])
        d.text((x + S(20), S(438)), 'Last command', font=F('m', S(14)), fill=T['ink2'])
        last = st['log'][-1] if st['log'] else None
        d.text((x + S(20), S(458)), PRETTY[last] if last else 'None yet', font=F('b', S(22)), fill=T['ink'] if last else T['ink3'])
        d.text((x + S(4), rb + S(14)), 'Wave off cancels at any time.', font=F('r', S(15)), fill=T['ink2'])
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


def tint(col, base, a):
    return tuple(int(round(base[i] * (1 - a) + col[i] * a)) for i in range(3))


def cv2_circle(img, c, r, col):
    import cv2
    cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
