"""
camera.py - Engine đọc camera: định danh theo instance_id + lớp Camera đa luồng.
================================================================================
- list_real_cameras()/find_index_by_instance(): map instance_id (ổn định theo
  cổng USB) -> index DSHOW hiện tại bằng cv2_enumerate_cameras.
- Camera: bọc 1 VideoCapture + 1 thread daemon RIÊNG (continuous-grab, trễ thấp),
  tự kết nối lại trong nền, có watchdog chống đứng hình.
- draw_label()/draw_index_badge(): vẽ nhãn + badge số lên cell.
"""

import cv2
import time
import threading

from config import (CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS, CAPTURE_FOURCC,
                    VIRTUAL_KEYWORDS, DISPLAY_FPS, RECONNECT_INTERVAL,
                    GRAB_FPS, GRAB_FAIL_LIMIT, SNAPSHOT_FLUSH)

# Serialize mở VideoCapture + enumerate trên DSHOW. Mở 6 cam song song dễ xung đột
# USB negotiate; enumerate song song (nhiều cam cùng reconnect lúc rút dây) trả
# index không nhất quán -> 2 cam map nhầm cùng index -> TRÙNG LUỒNG.
_OPEN_LOCK = threading.Lock()


# ============================================================
# LIỆT KÊ CAMERA THẬT + MAP INSTANCE_ID -> INDEX DSHOW
# ============================================================
def _norm(s):
    """Chuẩn hóa so khớp: chữ thường + chỉ giữ alphanumeric -> instance_id khớp
    được với path dù khác dấu \\ vs # và hoa/thường."""
    return "".join(c for c in str(s).lower() if c.isalnum())


def list_real_cameras():
    """Liệt kê camera THẬT (loại cam ảo theo VIRTUAL_KEYWORDS).
    Trả [(index, name, path), ...], hoặc None nếu thiếu lib."""
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
    """Tìm index DSHOW của camera có path khớp 'target' (instance_id/path).
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
# VẼ NHÃN + BADGE LÊN CELL (cv2, nhanh)
# ============================================================
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


# ============================================================
# LỚP ĐỌC CAMERA ĐA LUỒNG + TỰ KẾT NỐI LẠI
# ============================================================
class Camera:
    """Bọc 1 VideoCapture + 1 thread daemon RIÊNG. Thread liên tục grab() để xả
    buffer DSHOW (khung luôn tươi), chỉ retrieve()+render theo nhịp DISPLAY_FPS.
    Ghi vào self.cell của RIÊNG nó (gán atomic dưới GIL -> main đọc không cần
    lock). Tự mở lại trong thread của nó -> 1 cam mất/treo không ảnh hưởng cam khác."""

    def __init__(self, name, number, cell_size, src=None, target=None,
                 rotate180=False):
        self.src = src                          # index DSHOW (có thể tự resolve)
        self.target = target                    # instance_id/path để khớp ra index
        self.name = name
        self.number = number                    # số thứ tự 1..6 cho badge
        self.cell_w, self.cell_h = cell_size
        self.rotate180 = rotate180              # xoay ảnh 180° (cam lắp ngược)
        self.cap = None
        self.cell = None                        # ô render gần nhất (None=mất KN)
        self.running = False
        self.thread = None
        self._last_open_attempt = 0.0           # mốc lần cuối THỬ mở (cooldown)
        self._grab_fails = 0                    # đếm grab/retrieve lỗi liên tiếp
        self.last_frame_time = 0.0              # mốc grab() thành công gần nhất
        self._reset_requested = False           # watchdog đặt True để ép reset
        self.fps = 0.0                          # FPS quan sát (1/khoảng cách decode)
        self._last_decode_t = 0.0

    def _open(self):
        """Mở camera, trả True nếu được. In codec/độ phân giải THỰC TẾ để soi cam
        nào rớt về raw (YUYV) gây nghẽn.

        NỚI KHÓA: chỉ bước DÒ INDEX theo instance_id (enumerate) chạy trong
        _OPEN_LOCK để serialize (enumerate song song trả index không nhất quán ->
        TRÙNG LUỒNG). Còn lệnh VideoCapture MỞ NGOÀI khóa -> nhiều cam mở SONG SONG
        (nhanh hơn ở chế độ tuần tự nhiều nhóm). Đánh đổi: mở song song trên DShow
        thi thoảng trượt -> trả False, vòng chụp tự thử lại (không sập app); và nếu
        cắm/rút ngay trong lúc mở, ô có thể sai 1 nhịp rồi tự đúng lại lượt sau."""
        # 1) Resolve index theo instance_id DƯỚI khóa (chỉ enumerate, rất nhanh).
        if self.target:
            with _OPEN_LOCK:
                idx = find_index_by_instance(self.target, list_real_cameras() or [])
            if idx is None:
                return False
            self.src = idx
        if self.src is None:                 # không resolve được -> coi như chưa cắm
            return False
        # 2) Mở VideoCapture NGOÀI khóa (cho phép mở song song nhiều cam).
        cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        # Ép codec (CAPTURE_FOURCC) TRƯỚC khi set độ phân giải. YUY2=raw (nhẹ
        # CPU, nặng USB) / MJPG=nén (nặng CPU decode, nhẹ USB). Xem config.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAPTURE_FOURCC))
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
            return True
        cap.release()
        return False

    def _release_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _render_cell(self, frame):
        """Resize khung nguồn -> cell + vẽ nhãn + badge. Xoay 180° phần ẢNH nếu
        cam lắp ngược; nhãn/badge vẽ sau nên vẫn xuôi."""
        if self.rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        cell = cv2.resize(frame, (self.cell_w, self.cell_h),
                          interpolation=cv2.INTER_AREA)
        draw_label(cell, self.name, self.src, self.fps)
        draw_index_badge(cell, self.number)
        return cell

    def snapshot(self, flush=SNAPSHOT_FLUSH):
        """[nhóm tuần tự] Mở (nếu chưa) -> XẢ vài khung cũ trong buffer -> đọc 1
        khung -> render vào self.cell. KHÔNG tự release (orchestrator đóng sau khi
        giữ ảnh). Trả True nếu cập nhật được ô. Lỗi -> cell=None + nuốt exception
        để 1 cam hỏng không làm sập vòng tuần tự."""
        try:
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    self.cell = None
                    return False
            for _ in range(max(0, flush)):       # bỏ ảnh cũ kẹt trong buffer
                self.cap.grab()
            ok, frame = self.cap.read()          # grab + retrieve 1 khung tươi
            if not ok or frame is None:
                self._release_cap()
                self.cell = None
                return False
            self.cell = self._render_cell(frame)
            return True
        except Exception as e:
            print(f"[!] {self.name} (dev {self.src}) lỗi snapshot: {e}")
            self._release_cap()
            self.cell = None
            return False

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
        self._release_cap()

    def request_reset(self):
        """Watchdog (thread main) gọi khi cam đứng hình: đặt cờ + release cap từ
        NGOÀI để phá grab() đang treo. Thread cam sẽ xử lý cờ ở đầu vòng rồi tự mở lại."""
        self._reset_requested = True
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def _run(self):
        """Vòng đọc riêng của cam: grab() xả buffer (khung luôn tươi), chỉ
        retrieve()+render theo nhịp DISPLAY_FPS. Tự mở lại khi rớt (cooldown).
        GHÌM NHỊP vòng theo GRAB_FPS để diệt busy-spin (grab() trả về tức thì)."""
        decode_period = 1.0 / DISPLAY_FPS
        grab_interval = 1.0 / max(1, GRAB_FPS)
        next_decode = time.time()
        next_grab = time.time()
        while self.running:
            # (watchdog) Bị ép reset do đứng hình -> đóng cap & mở lại NGAY.
            if self._reset_requested:
                self._reset_requested = False
                self._release_cap()              # idempotent (cap có thể đã release)
                self.cell = None
                self.last_frame_time = time.time()   # debounce watchdog
                self._last_open_attempt = 0.0        # mở lại ngay, bỏ qua cooldown
                continue

            # (a) Đảm bảo cap mở; có cooldown để không thử mở lại dồn dập.
            if self.cap is None or not self.cap.isOpened():
                now = time.time()
                if now - self._last_open_attempt < RECONNECT_INTERVAL:
                    self.cell = None              # ô hiện "Kiểm tra lại kết nối"
                    time.sleep(0.1)               # tránh spin trong lúc chờ cooldown
                    continue
                self._last_open_attempt = now
                if not self._open():
                    self.cell = None
                    continue
                print(f"[+] {self.name} (dev {self.src}) đã kết nối.")
                self.last_frame_time = time.time()   # mốc khởi đầu cho watchdog
                self._grab_fails = 0
                next_decode = time.time()
                next_grab = time.time()

            # (b) grab() XẢ buffer; lỗi thoáng qua (rung USB) được dung sai, chỉ coi
            # mất kết nối sau GRAB_FAIL_LIMIT lần lỗi LIÊN TIẾP (tránh rớt oan).
            if not self.cap.grab():
                self._grab_fails += 1
                if self._grab_fails >= GRAB_FAIL_LIMIT:
                    print(f"[!] {self.name} (dev {self.src}) mất kết nối.")
                    self._release_cap()
                    self.cell = None
                    self._grab_fails = 0
                else:
                    time.sleep(grab_interval)        # chờ chút rồi thử lại, không spin
                continue
            self._grab_fails = 0
            self.last_frame_time = time.time()       # có khung mới -> watchdog yên tâm

            # (c) decode + render GIÃN theo DISPLAY_FPS để tiết chế CPU.
            now = time.time()
            if now >= next_decode:
                ok, frame = self.cap.retrieve()
                if not ok or frame is None:
                    self._grab_fails += 1
                    if self._grab_fails >= GRAB_FAIL_LIMIT:
                        print(f"[!] {self.name} (dev {self.src}) mất kết nối.")
                        self._release_cap()
                        self.cell = None
                        self._grab_fails = 0
                        continue
                    # dưới ngưỡng: bỏ khung này, xuống pacing rồi thử lại (không spin)
                else:
                    self._grab_fails = 0
                    if self._last_decode_t > 0.0:
                        dt = now - self._last_decode_t
                        if dt > 0:
                            self.fps = 1.0 / dt
                    self._last_decode_t = now
                    self.cell = self._render_cell(frame)
                    next_decode = now + decode_period

            # (d) PACING: ghìm vòng ~GRAB_FPS để diệt busy-spin (CPU 97% -> ~70%).
            next_grab += grab_interval
            sleep_for = next_grab - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_grab = time.time()              # tụt nhịp -> reset, không tích nợ


# ============================================================
# DÒ CAMERA THEO INDEX DSHOW THÔ (fallback CAMERA_INDICES, chạy khi setup)
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
