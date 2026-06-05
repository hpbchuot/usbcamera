"""
Multi-Camera Viewer - Hiển thị 6 webcam USB (720p) trên 1 màn hình
====================================================================
Chế độ MỖI CAM 1 THREAD RIÊNG (continuous-grab, trễ thấp):
- Mỗi cam có 1 thread daemon liên tục grab() để XẢ buffer DirectShow nên khung
  luôn TƯƠI (trễ ~1-2 frame thay vì 1-3s). DSHOW thường bỏ qua BUFFERSIZE=1 ->
  đọc liên tục là cách tin cậy để buffer luôn rỗng.
- Chỉ retrieve() (decode) + vẽ theo nhịp DISPLAY_FPS để tiết chế CPU; grab() xả
  buffer thì rẻ (không decode) nên vẫn nhẹ.
- Mỗi cam chỉ ghi vào Ô CỦA RIÊNG NÓ -> 1 cam mất KHÔNG làm ô cam khác đổi/mất
  hình; tự mở lại trong chính thread của nó nên không kẹt cam khác.
Tự kết nối lại trong nền:
    * Camera chưa cắm  -> ô đó hiện "Kiểm tra lại kết nối camera N".
    * Cắm vào lúc sau  -> ô đó tự hiện hình, không cần khởi động lại app.
    * Bị rút giữa chừng -> quay lại hiện dòng cảnh báo, các cam khác vẫn chạy.
- Ép codec MJPG để tiết kiệm băng thông USB; in codec/độ phân giải THỰC TẾ
  lúc mở để soi cam nào rớt về raw (YUYV) gây nghẽn.
- Chữ tiếng Việt vẽ bằng Pillow (cv2.putText không hỗ trợ dấu).

ĐỊNH DANH CAMERA THEO INSTANCE_ID (ổn định theo cổng USB vật lý):
- Mỗi CAM khớp theo instance_id (CAMERA_INSTANCE_IDS) bằng cv2_enumerate_cameras
  -> miễn nhiễm với việc index DSHOW bị đảo do có camera ảo (ManyCam) / cắm lại.
- Lấy instance_id:  python multi_camera_viewer.py discover

BỐ CỤC LƯỚI 3x2 (số chẵn HÀNG TRÊN, số lẻ HÀNG DƯỚI, theo cột):
        CAM2   CAM4   CAM6
        CAM1   CAM3   CAM5

Phím tắt:
    q hoặc ESC : thoát
    f          : bật/tắt toàn màn hình
    s          : chụp ảnh lưới hiện tại (lưu ra file .jpg)

Cài đặt: pip install opencv-python numpy pillow cv2-enumerate-cameras

Tác giả: viết cho Tys
"""

import cv2
import sys
import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

CAMERA_INSTANCE_IDS = [
    r"USB\VID_4C4A&PID_4A55&MI_00\6&17f9c0cf&0&0000",  # CAM 1 -> dưới-trái
    r"USB\VID_4C4A&PID_4A55&MI_00\6&540102a&0&0000",   # CAM 2 -> trên-trái
    r"USB\VID_4C4A&PID_4A55&MI_00\6&2e21298c&0&0000",  # CAM 3 -> dưới-giữa
    r"USB\VID_4C4A&PID_4A55&MI_00\6&1b6778e7&0&0000",  # CAM 4 -> trên-giữa
    r"USB\VID_4C4A&PID_4A55&MI_00\6&8adc842&0&0000",   # CAM 5 -> dưới-phải
    r"USB\VID_4C4A&PID_4A55&MI_00\6&a0be863&0&0000",   # CAM 6 -> trên-phải
]

# Từ khóa tên camera ẢO cần loại bỏ khi liệt kê (ManyCam, OBS...). So khớp
# không phân biệt hoa/thường.
VIRTUAL_KEYWORDS = ["manycam", "obs", "virtual", "xsplit",
                    "snap camera", "droidcam", "splitcam", "e2esoft", "iriun"]

# Fallback khi CAMERA_INSTANCE_IDS để trống ([]) hoặc thiếu thư viện enumerate:
# dùng index DSHOW thủ công cho từng ô 1..6 (dò bằng --detect).
CAMERA_INDICES = [0, 1, 2, 3, 4, 5]

# Các CAM (theo SỐ thứ tự 1..6) cần xoay khung hình 180° (cam lắp ngược).
# Mặc định CAM hàng dưới (1,3,5).
ROTATE_180_CAMS = [1, 3, 5]

# Độ phân giải bắt từ camera (nguồn). HD 720p 16:9.
# Hạ từ 1080p -> 720p để giảm băng thông USB + CPU giải mã MJPEG.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30

# Kích thước MỖI Ô hiển thị (giữ tỉ lệ 16:9).
# 640x360 x lưới 3x2 => cửa sổ 1920x720.
CELL_WIDTH = 640
CELL_HEIGHT = 360

# Bố cục lưới: 3 cột x 2 hàng = 6 ô
GRID_COLS = 3
GRID_ROWS = 2

# --- Chế độ MỖI CAM 1 THREAD RIÊNG (continuous-grab) ---
# Mỗi cam có 1 thread daemon liên tục grab() để XẢ buffer DirectShow (DSHOW
# thường bỏ qua CAP_PROP_BUFFERSIZE=1, đọc thưa sẽ dồn khung cũ gây trễ 1-3s).
# grab() rẻ (không decode); chỉ retrieve()+render theo nhịp DISPLAY_FPS để nhẹ CPU.
DISPLAY_FPS = 10            # nhịp decode + render mỗi cam (chỉnh cân CPU/độ mượt)
# Nhịp ghìm vòng grab() (= tốc độ camera sinh khung). QUAN TRỌNG: grab() của
# DSHOW thường KHÔNG block -> nếu vòng while không ghìm sẽ busy-spin chiếm trọn
# CPU. Ghìm về ~GRAB_FPS để vừa đủ xả buffer (trễ thấp) mà không spin.
GRAB_FPS = CAPTURE_FPS
# Số lần grab() lỗi LIÊN TIẾP trước khi coi là mất kết nối thật. Vì grab() có thể
# trả False tạm thời (chưa kịp khung mới do nhịp/sleep) -> không buông cap ngay,
# tránh ngắt/mở lại oan (gây dò DSHOW dồn dập -> dễ crash trên máy yếu).
GRAB_FAIL_LIMIT = 8
# Cooldown thử MỞ LẠI 1 camera đang mất (giây). Mở DirectShow một camera đã rớt
# có thể block ~1-3s; giãn ra để thread của cam đó không thử mở lại dồn dập.
# (Mỗi cam thread riêng nên dù có block cũng KHÔNG ảnh hưởng cam khác.)
RECONNECT_INTERVAL = 5.0

WINDOW_NAME = "Multi-Camera Viewer (6 CAM)"

# Khóa toàn cục: serialize việc mở VideoCapture trên DSHOW.
# Mở 6 cam song song dễ gây xung đột USB negotiate -> 1-2 cam fail lúc khởi động.
_OPEN_LOCK = threading.Lock()


# ============================================================
# LIỆT KÊ CAMERA THẬT + MAP INSTANCE_ID -> INDEX DSHOW
# ============================================================
# Dùng cv2_enumerate_cameras (pip install cv2-enumerate-cameras) để lấy
# (index, name, path) của mọi camera DSHOW. 'path' chứa instance_id của thiết bị
# nên ta khớp instance_id người dùng khai báo với path -> ra index hiện tại.
def _norm(s):
    """Chuẩn hóa chuỗi để so khớp: chữ thường + chỉ giữ ký tự alphanumeric.
    Nhờ vậy 'USB\\VID_4C4A...\\6&17f9c0cf&0&0000' khớp được với path dạng
    '\\\\?\\usb#vid_4c4a...#6&17f9c0cf&0&0000#{guid}\\global' (khác dấu \\ vs #)."""
    return "".join(c for c in str(s).lower() if c.isalnum())


def list_real_cameras():
    """Liệt kê camera THẬT (loại camera ảo theo VIRTUAL_KEYWORDS).
    Trả [(index, name, path), ...] theo thứ tự enumerate, hoặc None nếu thiếu lib."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except Exception as e:
        print(f"[!] Thiếu cv2-enumerate-cameras ({e}). "
              f"Cài: pip install cv2-enumerate-cameras")
        return None
    cams = []
    for info in enumerate_cameras(cv2.CAP_DSHOW):
        name = info.name or ""
        if any(k in name.lower() for k in VIRTUAL_KEYWORDS):
            continue
        cams.append((info.index, name, info.path or ""))
    return cams


def find_index_by_instance(target, cams):
    """Tìm index DSHOW của camera có path khớp 'target' (instance_id hoặc path).
    Khớp bằng _norm substring. Trả None nếu không thấy."""
    if not target or not cams:
        return None
    key = _norm(target)
    if not key:
        return None
    for index, _name, path in cams:
        if key in _norm(path):
            return index
    return None


# ============================================================
# VẼ CHỮ TIẾNG VIỆT (Unicode) BẰNG PILLOW
# ============================================================
_FONT_CACHE = {}


def _get_font(size):
    """Nạp font hỗ trợ tiếng Việt, cache lại theo size."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def put_text_vn_center(img, text, cy, size, color_bgr):
    """Vẽ chữ tiếng Việt canh giữa theo chiều ngang, tại tung độ cy."""
    font = _get_font(size)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.shape[1] - tw) // 2
    y = cy - th // 2
    # PIL dùng RGB nên đảo thứ tự màu từ BGR
    draw.text((x, y), text, font=font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# LỚP ĐỌC CAMERA ĐA LUỒNG + TỰ KẾT NỐI LẠI
# ============================================================
class Camera:
    """Bọc 1 VideoCapture + 1 thread daemon RIÊNG.

    Thread liên tục grab() để XẢ buffer DirectShow (giữ khung luôn tươi, trễ
    ~1-2 frame), chỉ retrieve()+render theo nhịp DISPLAY_FPS để nhẹ CPU. Mỗi cam
    ghi vào self.cell của RIÊNG nó (gán tham chiếu là atomic dưới GIL -> main đọc
    trực tiếp, không cần lock). Tự mở lại trong chính thread của nó -> 1 cam
    mất/treo KHÔNG ảnh hưởng cam khác.
    """

    def __init__(self, name, number, cell_size, src=None, target=None,
                 rotate180=False):
        self.src = src                          # index DSHOW (có thể tự resolve)
        self.target = target                    # instance_id/path để khớp ra index
        self.name = name
        self.number = number                    # số thứ tự 1..6 cho badge
        self.cell_w, self.cell_h = cell_size
        self.rotate180 = rotate180              # xoay ảnh 180° (cam lắp ngược)
        self.cap = None
        self.cell = None                        # ô đã render gần nhất (None=mất KN)
        self.running = False
        self.thread = None
        # Mốc lần cuối THỬ mở (cho cooldown reconnect) - tránh mở lại dồn dập.
        self._last_open_attempt = 0.0
        self._grab_fails = 0                    # đếm grab lỗi liên tiếp (dung sai)
        # FPS quan sát được (1 / khoảng cách giữa 2 lần decode của cam này)
        self.fps = 0.0
        self._last_decode_t = 0.0

    def _open(self):
        """Thử mở camera. Trả True nếu mở được.

        CHỈ enumerate (list_real_cameras) khi CHƯA biết index (lần đầu, hoặc sau
        khi mở hỏng nghi index đã đổi). Reconnect thông thường mở lại ĐÚNG index
        cũ, KHÔNG dò -> tránh 6 thread cùng dò DSHOW dồn dập (nguyên nhân dễ crash
        native). Toàn bộ (dò + mở) chạy trong _OPEN_LOCK để serialize an toàn."""
        with _OPEN_LOCK:
            # Dò index từ instance_id chỉ khi chưa biết.
            if self.src is None:
                if not self.target:
                    return False
                idx = find_index_by_instance(self.target, list_real_cameras() or [])
                if idx is None:
                    return False
                self.src = idx

            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
            # Ép MJPG TRƯỚC khi set độ phân giải để tránh nghẽn băng thông USB.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fc = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc = "".join(chr((fc >> 8 * k) & 0xFF) for k in range(4))
                print(f"[open] {self.name} (dev {self.src}): {aw}x{ah} {fourcc}")
                self.cap = cap
                self._grab_fails = 0
                return True
            cap.release()
            # Mở hỏng với index đã biết -> index có thể đã đổi (đảo/ManyCam) ->
            # buộc dò lại (enumerate) ở lần mở kế tiếp.
            if self.target:
                self.src = None
            return False

    def _release_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _render_cell(self, frame):
        """Resize khung nguồn -> cell + vẽ nhãn + badge.
        Xoay 180° phần ẢNH nếu cam lắp ngược; nhãn/badge vẽ sau nên vẫn xuôi."""
        if self.rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        cell = cv2.resize(frame, (self.cell_w, self.cell_h),
                          interpolation=cv2.INTER_AREA)
        draw_label(cell, self.name, self.src, self.fps)
        draw_index_badge(cell, self.number)
        return cell

    def start(self):
        """Bật thread daemon đọc của riêng cam này."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        # KHÔNG release cap ở đây: thread tự release trong finally của _run. Tránh
        # release CHÉO LUỒNG khi thread còn đang grab() -> dễ crash native. Thread
        # treo quá 2s sẽ được HĐH dọn khi tiến trình thoát (daemon).

    def _run(self):
        """Vòng đọc riêng của cam: grab() liên tục để xả buffer (khung luôn tươi),
        chỉ retrieve()+render theo nhịp DISPLAY_FPS. Tự mở lại khi rớt (cooldown).
        1 cam mất/treo chỉ ảnh hưởng thread này, không kẹt cam khác."""
        decode_period = 1.0 / DISPLAY_FPS
        grab_interval = 1.0 / max(1, GRAB_FPS)    # ghìm vòng grab ~ tốc độ sinh khung
        next_decode = time.time()
        next_grab = time.time()
        try:
            while self.running:
                # (a) Đảm bảo cap mở; có cooldown để không thử mở lại dồn dập.
                if self.cap is None or not self.cap.isOpened():
                    now = time.time()
                    if now - self._last_open_attempt < RECONNECT_INTERVAL:
                        self.cell = None          # ô hiện "Kiểm tra lại kết nối"
                        time.sleep(0.1)           # tránh spin trong lúc chờ cooldown
                        continue
                    self._last_open_attempt = now
                    if not self._open():
                        self.cell = None
                        continue
                    print(f"[+] {self.name} (dev {self.src}) đã kết nối.")
                    next_decode = time.time()
                    next_grab = time.time()       # reset nhịp grab, tránh burst

                # (b) grab() xả buffer. DUNG SAI: grab() có thể False tạm thời ->
                # chỉ buông cap sau GRAB_FAIL_LIMIT lần lỗi liên tiếp (tránh ngắt oan).
                try:
                    grabbed = self.cap.grab()
                except cv2.error:
                    grabbed = False
                if not grabbed:
                    self._grab_fails += 1
                    if self._grab_fails >= GRAB_FAIL_LIMIT:
                        print(f"[!] {self.name} (dev {self.src}) mất kết nối.")
                        self._release_cap()
                        self.cell = None
                        self._grab_fails = 0
                    else:
                        time.sleep(grab_interval)  # chờ khung kế, không spin
                    continue
                self._grab_fails = 0

                # (c) decode + render GIÃN theo DISPLAY_FPS để tiết chế CPU.
                now = time.time()
                if now >= next_decode:
                    try:
                        ok, frame = self.cap.retrieve()
                    except cv2.error:
                        ok, frame = False, None
                    if ok and frame is not None:
                        if self._last_decode_t > 0.0:
                            dt = now - self._last_decode_t
                            if dt > 0:
                                self.fps = 1.0 / dt
                        self._last_decode_t = now
                        self.cell = self._render_cell(frame)
                        next_decode = now + decode_period
                    # retrieve hỏng -> bỏ qua khung này; nếu cap chết thật, (b) sẽ
                    # đếm grab lỗi rồi buông. Không release ngay (tránh ngắt oan).

                # (d) GHÌM NHỊP vòng grab -> hết busy-spin (grab() DSHOW không block).
                next_grab += grab_interval
                sleep_for = next_grab - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_grab = time.time()       # tụt nhịp -> reset, không tích nợ
        finally:
            # Giải phóng cap trong CHÍNH thread của cam (an toàn, không chéo luồng).
            self._release_cap()


# ============================================================
# TIỆN ÍCH GHÉP LƯỚI
# ============================================================
def make_disconnected_cell(number):
    """Ô báo chưa kết nối, có dòng tiếng Việt canh giữa + badge số.

    PIL render text rất chậm (~5-15ms/cell) -> KHÔNG được gọi mỗi frame trong main
    loop. Hàm này được gọi MỘT LẦN lúc khởi động (xem build_disconnected_cells)
    để cache sẵn, main chỉ paste cell có sẵn vào canvas."""
    cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    msg = f"Kiểm tra lại kết nối camera {number}"
    cell = put_text_vn_center(cell, msg, CELL_HEIGHT // 2, 24, (60, 60, 240))
    draw_index_badge(cell, number)
    return cell


def build_disconnected_cells():
    """Render trước 6 ô 'mất kết nối' (kèm badge) để main chỉ việc paste."""
    return [make_disconnected_cell(i + 1) for i in range(GRID_COLS * GRID_ROWS)]


def draw_label(cell, name, dev_index, fps):
    """Nhãn nhỏ góc trên trái (ASCII, vẽ bằng cv2 cho nhanh)."""
    label = f"{name} (dev {dev_index}) | {fps:4.1f} FPS"
    cv2.rectangle(cell, (0, 0), (300, 26), (0, 0, 0), -1)
    cv2.putText(cell, label, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return cell


def draw_index_badge(cell, number):
    """Số thứ tự lớn (1-6) trong vòng tròn ở góc trên phải."""
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


def allocate_canvas():
    """Cấp phát MỘT LẦN canvas tổng cho lưới. Tránh np.hstack/vstack mỗi frame
    (mỗi lần gọi sẽ malloc + memcpy toàn bộ ảnh -> tốn CPU & gây jitter)."""
    return np.zeros((GRID_ROWS * CELL_HEIGHT,
                     GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)


def cell_position(i):
    """Vị trí (row, col) trong lưới của CAM số (i+1). Bố cục theo CỘT, số CHẴN ở
    HÀNG TRÊN, số LẺ ở HÀNG DƯỚI:
        CAM2  CAM4  CAM6   (row 0 - trên)
        CAM1  CAM3  CAM5   (row 1 - dưới)
    """
    col = i // 2
    row = 0 if (i + 1) % 2 == 0 else 1   # CAM chẵn -> trên; CAM lẻ -> dưới
    return row, col


def build_grid_into(canvas, cameras, disconnected_cells):
    """Paste cell mới nhất của từng camera vào canvas.

    Mỗi cam tự render cell trong thread của nó -> main thread chỉ memcpy (numpy
    slicing) -> rất nhẹ. cam.cell = None (cam chưa cắm / vừa rớt) -> thay bằng ô
    'Kiểm tra lại kết nối' đã cache sẵn. Vị trí ô theo cell_position()."""
    for i in range(GRID_COLS * GRID_ROWS):
        r, c = cell_position(i)
        y1, y2 = r * CELL_HEIGHT, (r + 1) * CELL_HEIGHT
        x1, x2 = c * CELL_WIDTH, (c + 1) * CELL_WIDTH

        cell = cameras[i].cell if i < len(cameras) else None
        if cell is None:
            cell = disconnected_cells[i]        # đã cache sẵn, không render lại
        canvas[y1:y2, x1:x2] = cell
    return canvas


# ============================================================
# TIỆN ÍCH DÒ CAMERA (chạy thủ công khi setup)
# ============================================================
def detect_cameras(max_index=10):
    print("Đang dò camera...")
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
                print(f"  - Tìm thấy camera tại index {i}")
            cap.release()
    print(f"Kết quả: {found}")
    return found


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================
def build_cameras():
    """Tạo danh sách Camera cho ô 1..N (chỉ số list = CAM (i+1)).

    - Ưu tiên CAMERA_INSTANCE_IDS: mỗi CAM khớp theo instance_id (ổn định theo
      cổng USB). In bảng đối chiếu instance_id -> index để kiểm tra.
    - Để trống -> fallback CAMERA_INDICES (index DSHOW thủ công)."""
    n_cells = GRID_COLS * GRID_ROWS
    cameras = []
    cams = list_real_cameras() if CAMERA_INSTANCE_IDS else None
    if CAMERA_INSTANCE_IDS and cams is None:
        print("[!] Không liệt kê được camera (thiếu lib) -> fallback CAMERA_INDICES.")
    if CAMERA_INSTANCE_IDS and cams is not None:
        targets = (list(CAMERA_INSTANCE_IDS) + [None] * n_cells)[:n_cells]
        print("Ánh xạ CAM -> instance_id -> index:")
        for i, target in enumerate(targets):
            idx = find_index_by_instance(target, cams or [])
            tail = (target or "").split("\\")[-1]
            print(f"  CAM {i + 1}: ...{tail or '(trống)'} -> "
                  f"{'index ' + str(idx) if idx is not None else 'CHƯA THẤY'}")
            cameras.append(Camera(name=f"CAM {i + 1}", number=i + 1,
                                  cell_size=(CELL_WIDTH, CELL_HEIGHT),
                                  target=target,
                                  rotate180=(i + 1) in ROTATE_180_CAMS))
        return cameras
    # Fallback: index thủ công
    srcs = (list(CAMERA_INDICES) + [None] * n_cells)[:n_cells]
    for i, src in enumerate(srcs):
        cameras.append(Camera(name=f"CAM {i + 1}", number=i + 1,
                              cell_size=(CELL_WIDTH, CELL_HEIGHT), src=src,
                              rotate180=(i + 1) in ROTATE_180_CAMS))
    return cameras


def main():
    print("Đang khởi tạo các camera...")
    cameras = build_cameras()
    # Mỗi cam tự chạy thread đọc riêng. Giãn cách lúc start để USB negotiate ổn
    # định (việc mở vẫn được _OPEN_LOCK serialize trong từng thread).
    for cam in cameras:
        cam.start()
        time.sleep(0.4)

    # Pre-allocate canvas + cache ô "mất kết nối" (Pillow text rất chậm,
    # không thể render mỗi frame trong main loop).
    canvas = allocate_canvas()
    disconnected_cells = build_disconnected_cells()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False
    print("Bắt đầu hiển thị. Phím: q/ESC thoát | f toàn màn hình | s chụp ảnh")

    # Nhịp ghép lưới + hiển thị (cell do thread cam tự cập nhật). Cap ~30fps để
    # cửa sổ mượt + waitKey được gọi đều (Windows không báo "Not Responding").
    frame_period = 1.0 / 30.0
    next_tick = time.time()

    try:
        while True:
            build_grid_into(canvas, cameras, disconnected_cells)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("s"):
                fname = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(fname, canvas)
                print(f"[+] Đã lưu {fname}")

            # Giữ nhịp display: ngủ phần thời gian còn dư trong period
            next_tick += frame_period
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Bị tụt nhịp -> reset mốc, không tích lũy nợ thời gian
                next_tick = time.time()
    finally:
        print("Đang đóng camera...")
        for cam in cameras:
            cam.stop()
        cv2.destroyAllWindows()
        print("Đã thoát.")


def discover():
    """In bảng index | name | path của các camera THẬT (đã lọc camera ảo) để
    đối chiếu, lấy instance_id (nằm trong path) điền vào CAMERA_INSTANCE_IDS."""
    cams = list_real_cameras()
    if cams is None:
        return
    if not cams:
        print("Không thấy camera thật nào (đã cắm cam chưa? có thể toàn cam ảo).")
        return
    print("\n========= CAMERA THẬT PHÁT HIỆN =========")
    for k, (index, name, path) in enumerate(cams, 1):
        print(f"[{k}] index={index}  name={name}")
        print(f"     path = {path}\n")
    print("=> Chép phần instance_id trong path (vd '6&17f9c0cf&0&0000') hoặc cả "
          "path vào CAMERA_INSTANCE_IDS theo thứ tự CAM 1..6.")


if __name__ == "__main__":
    # Liệt kê camera thật + path để lấy instance_id điền vào CAMERA_INSTANCE_IDS:
    #   python multi_camera_viewer.py discover
    if "discover" in sys.argv:
        discover()
    # Dò index DSHOW thô (fallback CAMERA_INDICES):
    #   python multi_camera_viewer.py --detect
    elif "--detect" in sys.argv:
        detect_cameras()
    else:
        main()
