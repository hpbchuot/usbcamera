"""
Multi-Camera Viewer - Hiển thị 6 webcam USB (720p) trên 1 màn hình
====================================================================
Chế độ CHỤP LUÂN PHIÊN THEO CẶP (thay cho stream 6 luồng liên tục):
- Mở cả 6 cam một lần và GIỮ mở; 1 thread scheduler quay vòng các cặp
  (1,2)(3,4)(5,6), mỗi chu kỳ chỉ grab 1 khung tươi từ 1 cặp.
- Trọn 1 vòng làm mới cả 6 ảnh trong ~REFRESH_PERIOD (mặc định 1s), cập nhật
  rải đều theo cặp nên mắt thấy mượt, không cảm giác trễ.
- Chỉ decode 2 cam/chu kỳ (thay vì 6×30fps) -> nhẹ CPU; đọc giãn cách ->
  giảm throughput thực tế trên bus USB (tránh nghẽn controller).
Tự kết nối lại trong nền:
    * Camera chưa cắm  -> ô đó hiện "Kiểm tra lại kết nối camera N".
    * Cắm vào lúc sau  -> ô đó tự hiện hình, không cần khởi động lại app.
    * Bị rút giữa chừng -> quay lại hiện dòng cảnh báo, các cam khác vẫn chạy.
- Ép codec MJPG để tiết kiệm băng thông USB; in codec/độ phân giải THỰC TẾ
  lúc mở để soi cam nào rớt về raw (YUYV) gây nghẽn.
- Ghép lưới 3x2, mỗi ô có số thứ tự lớn (1-6) + nhãn index thiết bị + FPS.
- Chữ tiếng Việt vẽ bằng Pillow (cv2.putText không hỗ trợ dấu).

Phím tắt:
    q hoặc ESC : thoát
    f          : bật/tắt toàn màn hình
    s          : chụp ảnh lưới hiện tại (lưu ra file .jpg)

Tác giả: viết cho Tys
"""

import cv2
import time
import threading
import numpy as np
import subprocess
import json
import re
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CẤU HÌNH - chỉnh ở đây
# ============================================================

# Index camera trên Windows (DirectShow). Nếu lệch index,
# chạy detect_cameras() bên dưới để dò index thực tế.
# Chú ý: khi AUTO_DETECT_PORTS=True và SORT_BY_PORT=True,
# thứ tự này sẽ tự được sắp xếp lại theo port vật lý.
CAMERA_INDICES = [0, 1, 2, 3, 4, 5]

# --- Tính năng nhận diện port USB vật lý (Device Manager) ---
# True: dò port USB khi khởi động và hiển thị "Port_#XXXX.Hub_#XXXX" trên ô.
AUTO_DETECT_PORTS = True
# True: sắp xếp lại thứ tự slot theo port (port nhỏ nhất = slot 1,...).
# Giúp slot luôn cố định với vị trí cắm vật lý dù unplug/replug.
SORT_BY_PORT = True

# Độ phân giải bắt từ camera (nguồn). HD 720p 16:9.
# Hạ từ 1080p -> 720p để giảm băng thông USB + CPU giải mã MJPEG.
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30

# Kích thước MỖI Ô hiển thị (giữ tỉ lệ 16:9).
# 640x360 x lưới 3x2 => cửa sổ 1920x720.
CELL_WIDTH = 640
CELL_HEIGHT = 360

# Bố cục lưới: 3 cột x 2 hàng = 6 ô
GRID_COLS = 3
GRID_ROWS = 2

# --- Chế độ chụp luân phiên theo cặp ---
# Thay vì stream 6 luồng liên tục, ta giữ cả 6 cam mở rồi mỗi chu kỳ chỉ grab
# 1 khung từ 1 cặp, quay vòng các cặp -> làm mới cả 6 ảnh trong ~REFRESH_PERIOD.
# Lợi ích: chỉ decode 2 cam/chu kỳ (nhẹ CPU) + đọc giãn cách (nhẹ băng thông bus).
REFRESH_PERIOD = 1.0                    # trọn 1 vòng làm mới cả 6 ảnh (giây)
CAMERA_PAIRS = [(0, 1), (2, 3), (4, 5)]  # ghép cặp cố định theo vị trí ô
# Số khung cũ cần grab bỏ trước khi lấy khung tươi (xả buffer khi cam vừa idle).
FLUSH_GRABS = 2
# Cell cũ hơn ngưỡng này -> coi như mất kết nối, ô hiện cảnh báo (không đông cứng).
STALE_CELL_TIMEOUT = REFRESH_PERIOD * 5

WINDOW_NAME = "Multi-Camera Viewer (6 CAM)"

# Khóa toàn cục: serialize việc mở VideoCapture trên DSHOW.
# Mở 6 cam song song dễ gây xung đột USB negotiate -> 1-2 cam fail lúc khởi động.
_OPEN_LOCK = threading.Lock()


# ============================================================
# DÒ PORT USB VẬT LÝ (Device Manager)
# ============================================================

def scan_usb_camera_ports():
    """
    Lấy danh sách camera DirectShow cùng port USB vật lý (Port_#XXXX.Hub_#XXXX).

    Dùng registry key DirectShow video capture class để đảm bảo thứ tự index
    trả về KHỚP ĐÚNG với index OpenCV/cv2.VideoCapture(i).

    Trả về list[dict] với keys: index (int), name (str), port (str).
    Nếu lỗi hoặc không tìm được -> trả về [].
    """
    ps = r"""
$guid = '{65E8773D-8F56-11D0-A3B9-00A0C9223196}'
$base = "HKLM:\SYSTEM\CurrentControlSet\Control\DeviceClasses\$guid"
if (-not (Test-Path $base)) { Write-Output '[]'; exit }

$result = @()
$idx = 0
Get-ChildItem $base | Sort-Object PSChildName | ForEach-Object {
    # PSChildName dạng: ##?#USB#VID_...#...#{guid}  hoặc ##?#PCI#...
    $raw = $_.PSChildName
    # Chuyển sang InstanceId: bỏ '##?#' đầu, bỏ '#{guid}' cuối, thay '#' bằng '\'
    $inst = $raw -replace '^##\?#','' -replace '#\{[0-9a-fA-F\-]+\}$','' -replace '#','\'

    $fname = ''
    $loc   = ''
    try { $fname = (Get-PnpDeviceProperty -InstanceId $inst -KeyName 'DEVPKEY_Device_FriendlyName' -EA Stop).Data } catch {}
    if (-not $fname) {
        try { $fname = (Get-PnpDevice -InstanceId $inst -EA Stop).FriendlyName } catch {}
    }
    try {
        $par = (Get-PnpDeviceProperty -InstanceId $inst -KeyName 'DEVPKEY_Device_Parent' -EA Stop).Data
        $loc = (Get-PnpDeviceProperty -InstanceId $par  -KeyName 'DEVPKEY_Device_LocationInfo' -EA Stop).Data
    } catch {}
    if (-not $loc) {
        try { $loc = (Get-PnpDeviceProperty -InstanceId $inst -KeyName 'DEVPKEY_Device_LocationInfo' -EA Stop).Data } catch {}
    }

    $result += [PSCustomObject]@{ index=$idx; name=if($fname){$fname}else{"Camera $idx"}; port="$loc" }
    $idx++
}
$result | ConvertTo-Json -Compress
"""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=25,
        )
        out = r.stdout.strip()
        if not out or out == "[]":
            return []
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        return [{"index": d["index"], "name": d.get("name", ""), "port": d.get("port", "")}
                for d in data]
    except Exception as exc:
        print(f"[warn] scan_usb_camera_ports: {exc}")
        return []


def _port_sort_key(port_str):
    """Chuẩn hóa Port_#0004.Hub_#0003 thành tuple(4, 3) để sort số, không sort chuỗi."""
    nums = re.findall(r"\d+", port_str or "")
    return tuple(int(n) for n in nums)


def _short_port(port_str):
    """Port_#0004.Hub_#0003 -> 'P4.H3' cho nhãn ngắn gọn."""
    nums = re.findall(r"\d+", port_str or "")
    if len(nums) >= 2:
        return f"P{int(nums[0])}.H{int(nums[1])}"
    return port_str or ""


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
    """Bọc 1 VideoCapture. KHÔNG tự chạy thread đọc liên tục.

    Mỗi cam được mở 1 lần và GIỮ mở; scheduler chủ động gọi grab_fresh_cell()
    theo lịch luân phiên để lấy 1 khung tươi rồi render thành cell (resize +
    nhãn + badge). Tự mở lại nếu cam bị rớt/chưa cắm.
    """

    def __init__(self, src, name, number, cell_size, port=""):
        self.src = src
        self.name = name
        self.number = number                    # số thứ tự 1..6 cho badge
        self.port = port                        # "Port_#XXXX.Hub_#XXXX" từ Device Manager
        self.cell_w, self.cell_h = cell_size
        self.cap = None
        self._fail = 0
        # FPS quan sát được (1 / khoảng cách giữa 2 lần grab thành công của cam này)
        self.fps = 0.0
        self._last_grab_t = 0.0

    def _open(self):
        """Thử mở camera. Trả True nếu mở được. In ra codec/độ phân giải THỰC TẾ
        được negotiate để soi cam nào rớt về raw (YUYV) gây nghẽn băng thông."""
        # Serialize việc mở camera để tránh xung đột USB negotiate khi nhiều
        # cam cùng được mở trên DSHOW một lúc.
        with _OPEN_LOCK:
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
        """Resize khung nguồn -> cell + vẽ nhãn + badge."""
        cell = cv2.resize(frame, (self.cell_w, self.cell_h),
                          interpolation=cv2.INTER_AREA)
        draw_label(cell, self.name, self.src, self.fps, self.port)
        draw_index_badge(cell, self.number)
        return cell

    def grab_fresh_cell(self):
        """Lấy 1 khung TƯƠI và render thành cell. Trả None nếu cam chưa sẵn sàng.

        - Tự mở (lại) nếu cap chưa mở -> hỗ trợ chưa cắm / cắm lại nóng.
        - grab() bỏ FLUSH_GRABS khung cũ còn kẹt trong buffer (do cam vừa idle
          giữa 2 lần được đọc), rồi retrieve() lấy khung mới nhất.
        """
        if self.cap is None or not self.cap.isOpened():
            if not self._open():
                return None
            print(f"[+] {self.name} (dev {self.src}) đã kết nối.")
            self._fail = 0

        for _ in range(FLUSH_GRABS):
            self.cap.grab()
        ret, frame = self.cap.retrieve()
        if not ret or frame is None:
            self._fail += 1
            if self._fail > 3:
                print(f"[!] {self.name} (dev {self.src}) mất kết nối.")
                self._release_cap()
            return None

        self._fail = 0
        now = time.time()
        if self._last_grab_t > 0.0:
            dt = now - self._last_grab_t
            if dt > 0:
                self.fps = 1.0 / dt
        self._last_grab_t = now
        return self._render_cell(frame)

    def stop(self):
        self._release_cap()


# ============================================================
# SCHEDULER - QUAY VÒNG CHỤP ẢNH THEO TỪNG CẶP CAMERA
# ============================================================
class PollingScheduler:
    """1 thread daemon quay vòng các cặp camera. Mỗi cặp: grab 1 khung tươi từ
    từng cam trong cặp -> render cell -> ghi vào kho dùng chung (cells/cell_ts).

    Nhịp: mỗi cặp được cấp ~REFRESH_PERIOD / số_cặp giây, nên trọn 1 vòng (cả 6
    ảnh) hoàn tất trong ~REFRESH_PERIOD. Cập nhật rải đều theo cặp -> mượt mắt.

    Vì chỉ decode 2 cam mỗi chu kỳ (thay vì 6×30fps) nên CPU nhẹ hẳn; đọc giãn
    cách cũng giảm throughput thực tế trên bus USB."""

    def __init__(self, cameras, pairs=CAMERA_PAIRS, period=REFRESH_PERIOD):
        self.cameras = cameras
        self.pairs = pairs
        self.period = period
        # Kho dùng chung: gán phần tử list là atomic dưới GIL nên main thread đọc
        # trực tiếp không cần lock.
        self.cells = [None] * len(cameras)      # cell đã render gần nhất
        self.cell_ts = [0.0] * len(cameras)     # thời điểm cập nhật cell
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def _run(self):
        slot = self.period / max(1, len(self.pairs))   # thời lượng mỗi cặp
        while self.running:
            for pair in self.pairs:
                t0 = time.time()
                # Đọc lần lượt 2 cam trong cặp (tránh 2 read song song đụng băng
                # thông nếu chung controller). Vẫn đủ nhanh trong slot.
                for i in pair:
                    if not self.running:
                        return
                    cell = self.cameras[i].grab_fresh_cell()
                    if cell is not None:
                        self.cells[i] = cell
                        self.cell_ts[i] = time.time()
                # Giãn nhịp để mỗi cặp chiếm đúng ~slot giây
                dt = time.time() - t0
                if dt < slot:
                    time.sleep(slot - dt)


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


def draw_label(cell, name, dev_index, fps, port=""):
    """Nhãn nhỏ góc trên trái (ASCII, vẽ bằng cv2 cho nhanh).
    Dòng 1: tên + FPS.  Dòng 2 (nếu có port): port vật lý dạng 'P4.H3 (dev 0)'.
    """
    short = _short_port(port)
    line1 = f"{name} | {fps:4.1f} FPS"
    if short:
        line2 = f"{short}  (dev {dev_index})"
        cv2.rectangle(cell, (0, 0), (320, 46), (0, 0, 0), -1)
        cv2.putText(cell, line1, (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(cell, line2, (6, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 210, 255), 1)
    else:
        line1 = f"{name} (dev {dev_index}) | {fps:4.1f} FPS"
        cv2.rectangle(cell, (0, 0), (300, 26), (0, 0, 0), -1)
        cv2.putText(cell, line1, (6, 18),
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


def build_grid_into(canvas, scheduler, disconnected_cells):
    """Paste cell mới nhất của từng camera (do scheduler render) vào canvas.

    Cell đã được scheduler resize + vẽ nhãn + badge sẵn -> main thread chỉ làm
    memcpy (numpy slicing) -> rất nhẹ. Cell quá cũ (cam treo/mất kết nối) ->
    thay bằng ô 'Kiểm tra lại kết nối' đã cache sẵn."""
    now = time.time()
    for i in range(GRID_COLS * GRID_ROWS):
        r = i // GRID_COLS
        c = i % GRID_COLS
        y1, y2 = r * CELL_HEIGHT, (r + 1) * CELL_HEIGHT
        x1, x2 = c * CELL_WIDTH, (c + 1) * CELL_WIDTH

        cell = None
        if i < len(scheduler.cells):
            cell = scheduler.cells[i]
            if cell is None or now - scheduler.cell_ts[i] > STALE_CELL_TIMEOUT:
                cell = None
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
def main():
    # --- Dò port USB ---
    port_map = {}   # {directshow_index: "Port_#XXXX.Hub_#XXXX"}
    cam_indices = list(CAMERA_INDICES)

    if AUTO_DETECT_PORTS:
        print("Đang dò port USB từ Device Manager...")
        port_info = scan_usb_camera_ports()
        if port_info:
            print("\n=== Bảng ánh xạ Camera -> Port USB ===")
            for p in port_info:
                port_map[p["index"]] = p["port"]
                short = _short_port(p["port"])
                print(f"  dev {p['index']:>2}: {p['name'][:40]:<40}  {short}  ({p['port']})")

            if SORT_BY_PORT:
                # Sắp lại thứ tự index theo port vật lý (port nhỏ nhất = slot 1).
                # Chỉ lấy tối đa GRID_COLS*GRID_ROWS cam; bổ sung bằng CAMERA_INDICES nếu thiếu.
                sorted_indices = sorted(
                    [p["index"] for p in port_info],
                    key=lambda i: _port_sort_key(port_map.get(i, ""))
                )
                cam_indices = sorted_indices[: GRID_COLS * GRID_ROWS]
                # Nếu port_info trả về ít hơn số ô, bổ sung từ CAMERA_INDICES
                extra = [i for i in CAMERA_INDICES if i not in cam_indices]
                cam_indices += extra[: GRID_COLS * GRID_ROWS - len(cam_indices)]
                print(f"\nThứ tự slot (theo port): {cam_indices}")
            print()
        else:
            print("[warn] Không lấy được thông tin port USB; dùng thứ tự CAMERA_INDICES.\n")

    print("Đang khởi tạo các camera...")
    cameras = []
    for idx, cam_index in enumerate(cam_indices):
        cam = Camera(cam_index,
                     name=f"CAM {idx + 1}",
                     number=idx + 1,
                     cell_size=(CELL_WIDTH, CELL_HEIGHT),
                     port=port_map.get(cam_index, ""))
        # Mở trước, cách nhau chút để USB negotiate băng thông ổn định khi giữ
        # cả 6 cam mở đồng thời. Cam chưa cắm vẫn ok -> scheduler tự mở lại sau.
        cam._open()
        cameras.append(cam)
        time.sleep(0.5)

    # Scheduler quay vòng chụp theo cặp -> làm mới cả 6 ảnh mỗi ~REFRESH_PERIOD
    scheduler = PollingScheduler(cameras).start()

    # Pre-allocate canvas + cache ô "mất kết nối" (Pillow text rất chậm,
    # không thể render mỗi frame trong main loop).
    canvas = allocate_canvas()
    disconnected_cells = build_disconnected_cells()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False
    print("Bắt đầu hiển thị. Phím: q/ESC thoát | f toàn màn hình | s chụp ảnh")

    # Nhịp hiển thị nhẹ: ảnh chỉ đổi mỗi ~REFRESH_PERIOD nên không cần vẽ 30fps,
    # nhưng vẫn cần gọi waitKey đều để Windows không báo "Not Responding".
    frame_period = 1.0 / 30.0
    next_tick = time.time()

    try:
        while True:
            build_grid_into(canvas, scheduler, disconnected_cells)
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
        scheduler.stop()
        for cam in cameras:
            cam.stop()
        cv2.destroyAllWindows()
        print("Đã thoát.")


if __name__ == "__main__":
    # Muốn dò index camera, mở dòng dưới:
    # detect_cameras()
    main()
