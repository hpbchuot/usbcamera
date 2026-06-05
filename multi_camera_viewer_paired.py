"""
Multi-Camera Viewer - Chụp từng CẶP mỗi 0.5s, HARDCODE cổng USB
================================================================
Nhịp chụp: cứ mỗi PAIR_INTERVAL (0.5s) chụp 1 CẶP camera, lần lượt theo cột:
    (1,2) -> 0.5s -> (3,4) -> 0.5s -> (5,6) -> 0.5s -> lặp lại
Một lần chụp (1 cặp) diễn ra mỗi 0.5s; mỗi cặp làm mới sau 1 vòng (~1.5s).

Bố cục (đánh số theo CỘT):
    [1] [3] [5]   (hàng trên)
    [2] [4] [6]   (hàng dưới)

HARDCODE CỔNG USB:
    CAMERA_PATHS giữ 6 device path (mỗi path = 1 cổng USB vật lý). Đây là nguồn
    chính: CAM 1 luôn là camera ở cổng CAMERA_PATHS[0], v.v... Lấy path bằng:
        python multi_camera_viewer_paired.py discover
    rồi dán vào CAMERA_PATHS theo đúng thứ tự CAM 1..6.
    (Chưa điền đủ 6 -> tạm chạy theo index, có cảnh báo.)

Giữ nguyên: kiểm tra kết nối + tự mở lại (có cooldown), bỏ qua webcam ảo.
Camera giữ mở giữa các lần chụp (mở lại DirectShow rất chậm), chỉ đọc thưa.

Phím: q/ESC thoát | f toàn màn hình | s chụp ảnh lưới
Cài: pip install opencv-python numpy pillow cv2-enumerate-cameras pyinstaller
"""

import sys
import cv2
import time
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CẤU HÌNH
# ============================================================
# >>> HARDCODE CỔNG USB <<< điền 6 device path theo thứ tự CAM 1..6.
# Nhớ để chữ r trước dấu nháy (raw string) vì path có dấu \.
PORT_LABEL_MAP = {
    0: "Port_#0001.Hub_#0003",
    1: "Port_#0002.Hub_#0003",
    2: "Port_#0003.Hub_#0003",
    3: "Port_#0004.Hub_#0003",
    4: "Port_#0005.Hub_#0003",
    5: "Port_#0006.Hub_#0003",
}
CAMERA_INDICES = [0, 1, 2, 3, 4, 5]   # fallback khi CAMERA_PATHS chưa đủ

VIRTUAL_KEYWORDS = ["manycam", "obs", "virtual", "xsplit",
                    "snap camera", "droidcam", "splitcam", "e2esoft", "iriun"]

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30

CELL_WIDTH = 640
CELL_HEIGHT = 360
GRID_COLS = 3
GRID_ROWS = 2

PAIR_INTERVAL = 0.2          # giây: mỗi 0.5s chụp 1 cặp
RECONNECT_INTERVAL = 2.0     # cooldown thử mở lại camera mất kết nối
FLUSH_GRABS = 3              # số lần grab() để lấy frame mới (rẻ, không decode)

WINDOW_NAME = "Multi-Camera Viewer (cap doi - 0.5s)"


# ============================================================
# VẼ CHỮ TIẾNG VIỆT
# ============================================================
_FONT_CACHE = {}


def _get_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            _FONT_CACHE[size] = ImageFont.truetype(path, size)
            return _FONT_CACHE[size]
        except Exception:
            continue
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def put_text_vn_center(img, text, cy, size, color_bgr):
    font = _get_font(size)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = (img.shape[1] - tw) // 2, cy - th // 2
    draw.text((x, y), text, font=font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# LIỆT KÊ / MAP CAMERA THEO CỔNG USB
# ============================================================
def list_real_cameras():
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception as e:
        print(f"[!] Thiếu cv2-enumerate-cameras: {e}")
        return None
    cams = []
    for info in enumerate_cameras(cv2.CAP_DSHOW):
        name = (info.name or "")
        if any(k in name.lower() for k in VIRTUAL_KEYWORDS):
            continue
        cams.append((info.index, name, info.path))
    return cams


def find_index_by_path(target_path, cams=None):
    if cams is None:
        cams = list_real_cameras() or []
    for index, _name, path in cams:
        if path == target_path:
            return index
    return None


def resolve_camera_indices():
    cams = list_real_cameras()
    if cams:
        return [c[0] for c in cams][:GRID_COLS * GRID_ROWS] or CAMERA_INDICES
    return CAMERA_INDICES


# ============================================================
# CAMERA (chụp theo yêu cầu)
# ============================================================
class Camera:
    def __init__(self, name, slot, src=None, target_path=None):
        self.name = name
        self.slot = slot
        self.target_path = target_path
        self.src = src if src is not None else -1
        self.cap = None
        self.connected = False
        self.last_capture_time = None
        self._last_open_attempt = 0.0

    def _resolve_src(self):
        if self.target_path:
            idx = find_index_by_path(self.target_path)
            if idx is None:
                return False
            self.src = idx
        return self.src is not None and self.src >= 0

    def ensure_open(self):
        if self.cap is not None and self.cap.isOpened():
            return True
        now = time.time()
        if now - self._last_open_attempt < RECONNECT_INTERVAL:
            return False
        self._last_open_attempt = now

        if not self._resolve_src():
            return False
        cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            self.cap = cap
            print(f"[+] {self.name} (dev {self.src}) đã kết nối.")
            return True
        cap.release()
        return False

    def snapshot(self):
        if not self.ensure_open():
            self.connected = False
            return None
        for _ in range(FLUSH_GRABS):
            self.cap.grab()
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            print(f"[!] {self.name} (dev {self.src}) mất kết nối.")
            self.release()
            self.connected = False
            return None
        self.connected = True
        self.last_capture_time = datetime.now()
        return cv2.resize(frame, (CELL_WIDTH, CELL_HEIGHT))

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ============================================================
# VẼ Ô / LƯỚI
# ============================================================
_PLACEHOLDER_CACHE = {}


def draw_label(cell, name, dev_index, ts):
    t = ts.strftime("%H:%M:%S") if ts else "--:--:--"
    label = f"{name} (dev {dev_index}) | {t}"
    cv2.rectangle(cell, (0, 0), (300, 26), (0, 0, 0), -1)
    cv2.putText(cell, label, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return cell


def draw_index_badge(cell, number):
    h, w = cell.shape[:2]
    r = 28
    cx, cy = w - r - 12, r + 12
    cv2.circle(cell, (cx, cy), r, (0, 0, 0), -1)
    cv2.circle(cell, (cx, cy), r, (0, 255, 255), 2)
    text = str(number)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(cell, text, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return cell


def get_disconnected_cell(number):
    if number in _PLACEHOLDER_CACHE:
        return _PLACEHOLDER_CACHE[number].copy()
    cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    cell = put_text_vn_center(cell, f"Kiểm tra lại kết nối camera {number}",
                              CELL_HEIGHT // 2, 24, (60, 60, 240))
    cell = draw_index_badge(cell, number)
    _PLACEHOLDER_CACHE[number] = cell
    return cell.copy()


def build_grid(cells):
    """cells[i] ứng với camera index i. Sắp theo CỘT: i//GRID_ROWS = cột, i%GRID_ROWS = hàng."""
    grid_rc = [[None] * GRID_COLS for _ in range(GRID_ROWS)]
    for i, cell in enumerate(cells):
        r, c = i % GRID_ROWS, i // GRID_ROWS
        grid_rc[r][c] = cell
    rows_img = [np.hstack(grid_rc[r]) for r in range(GRID_ROWS)]
    return np.vstack(rows_img)


# ============================================================
# PHÍM & CHỜ CÓ PHẢN HỒI
# ============================================================
def handle_key(key, state):
    if key in (ord("q"), 27):
        return True
    if key == ord("f"):
        state["fullscreen"] = not state["fullscreen"]
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if state["fullscreen"] else cv2.WINDOW_NORMAL)
    elif key == ord("s"):
        fname = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(fname, state["grid"])
        print(f"[+] Đã lưu {fname}")
    return False


def pump_until(deadline, state):
    while time.time() < deadline:
        key = cv2.waitKey(20) & 0xFF
        if handle_key(key, state):
            return True
    return False


# ============================================================
# DỰNG DANH SÁCH CAMERA (ưu tiên hardcode cổng USB)
# ============================================================
def build_cameras():
    n = GRID_COLS * GRID_ROWS
    valid_paths = [p for p in CAMERA_PATHS if isinstance(p, str) and p.strip()]
    if len(valid_paths) >= n:
        print("Map CỐ ĐỊNH theo cổng USB (CAMERA_PATHS hardcode).")
        if list_real_cameras() is None:
            print("[!] Cần cv2-enumerate-cameras để tra path -> index.")
        return [Camera(f"CAM {i + 1}", i + 1, target_path=CAMERA_PATHS[i])
                for i in range(n)]

    print("[!] CAMERA_PATHS chưa điền đủ 6 cổng.")
    print("    Chạy:  python %s discover   để lấy path, rồi dán vào CAMERA_PATHS."
          % sys.argv[0])
    print("    Tạm thời chạy theo index để bạn còn xem được.")
    indices = resolve_camera_indices()
    print(f"    Index tạm: {indices}")
    return [Camera(f"CAM {i + 1}", i + 1, src=idx)
            for i, idx in enumerate(indices[:n])]


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================
def main():
    cameras = build_cameras()
    n = len(cameras)

    cells = [get_disconnected_cell(i + 1) for i in range(GRID_COLS * GRID_ROWS)]

    # Các cặp theo CỘT: (1,2),(3,4),(5,6)
    steps = [tuple(c * GRID_ROWS + r for r in range(GRID_ROWS))
             for c in range(GRID_COLS)]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    state = {"fullscreen": False, "grid": build_grid(cells)}
    print("Phím: q/ESC thoát | f toàn màn hình | s chụp ảnh")
    print(f"Mỗi {PAIR_INTERVAL:.1f}s chụp 1 cặp, lần lượt {steps}.")

    try:
        while True:
            for pair in steps:
                t0 = time.time()

                for cam_idx in pair:
                    if cam_idx >= n:
                        continue
                    cam = cameras[cam_idx]
                    frame = cam.snapshot()
                    if frame is not None:
                        cell = draw_label(frame, cam.name, cam.src, cam.last_capture_time)
                        cell = draw_index_badge(cell, cam.slot)
                        cells[cam_idx] = cell
                    else:
                        cells[cam_idx] = get_disconnected_cell(cam.slot)

                state["grid"] = build_grid(cells)
                cv2.imshow(WINDOW_NAME, state["grid"])
                if handle_key(cv2.waitKey(1) & 0xFF, state):
                    raise KeyboardInterrupt

                # Chờ đủ 0.5s kể từ lúc bắt đầu cặp này (CPU nghỉ phần còn lại)
                if pump_until(t0 + PAIR_INTERVAL, state):
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        print("Đang đóng camera...")
        for c in cameras:
            c.release()
        cv2.destroyAllWindows()
        print("Đã thoát.")


def discover():
    cams = list_real_cameras()
    if not cams:
        print("Không liệt kê được camera (cài: pip install cv2-enumerate-cameras).")
        return
    print("\n========= CAMERA PHÁT HIỆN =========")
    for k, (index, name, path) in enumerate(cams, 1):
        print(f"[{k}] index={index}  name={name}")
        print(f"     path = {path}\n")
    print("=> Dán các path theo thứ tự mong muốn vào CAMERA_PATHS (CAM 1..6).")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "discover":
        discover()
    else:
        main()
