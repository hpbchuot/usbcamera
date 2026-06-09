"""
multi_camera_viewer.py - UI ghép lưới + main loop + CLI (entry point).
======================================================================
Hiển thị 6 webcam USB trên lưới 3x2. Xem mô tả kiến trúc đầy đủ trong config.py.

Chạy:
    python multi_camera_viewer.py            # mở viewer
    python multi_camera_viewer.py discover   # liệt kê cam thật + path (lấy instance_id)
    python multi_camera_viewer.py --detect   # dò index DSHOW thô (fallback)

Phím tắt:  q/ESC thoát | f toàn màn hình | s chụp ảnh lưới (.jpg)
           R làm mới (đóng & mở lại) toàn bộ 6 camera
"""

import cv2
import sys
import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from config import (CAMERA_INSTANCE_IDS, CAMERA_INDICES, ROTATE_180_CAMS,
                    CELL_WIDTH, CELL_HEIGHT, GRID_COLS, GRID_ROWS,
                    FREEZE_TIMEOUT, WINDOW_NAME,
                    SEQUENTIAL_GROUPS, SEQ_DWELL)
from camera import (Camera, list_real_cameras, find_index_by_instance,
                    draw_index_badge, detect_cameras)


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
    draw.text((x, y), text, font=font,            # PIL dùng RGB nên đảo BGR
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# TIỆN ÍCH GHÉP LƯỚI
# ============================================================
def make_disconnected_cell(number):
    """Ô báo chưa kết nối (dòng tiếng Việt canh giữa + badge số). PIL render text
    rất chậm (~5-15ms) -> cache MỘT LẦN lúc khởi động, main chỉ paste."""
    cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    msg = f"Kiểm tra lại kết nối camera {number}"
    cell = put_text_vn_center(cell, msg, CELL_HEIGHT // 2, 24, (60, 60, 240))
    draw_index_badge(cell, number)
    return cell


def build_disconnected_cells():
    """Render trước 6 ô 'mất kết nối' (kèm badge) để main chỉ việc paste."""
    return [make_disconnected_cell(i + 1) for i in range(GRID_COLS * GRID_ROWS)]


def allocate_canvas():
    """Cấp phát MỘT LẦN canvas tổng cho lưới. Tránh np.hstack/vstack mỗi frame
    (mỗi lần gọi sẽ malloc + memcpy toàn bộ ảnh -> tốn CPU & gây jitter)."""
    return np.zeros((GRID_ROWS * CELL_HEIGHT,
                     GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)


def cell_position(i):
    """Vị trí (row, col) của CAM số (i+1). Bố cục theo CỘT, số CHẴN hàng TRÊN,
    số LẺ hàng DƯỚI:
        CAM2  CAM4  CAM6   (row 0 - trên)
        CAM1  CAM3  CAM5   (row 1 - dưới)
    """
    col = i // 2
    row = 0 if (i + 1) % 2 == 0 else 1   # CAM chẵn -> trên; CAM lẻ -> dưới
    return row, col


def build_grid_into(canvas, cameras, disconnected_cells):
    """Paste cell mới nhất của từng camera vào canvas (main chỉ memcpy, rất nhẹ).
    cam.cell = None (chưa cắm / vừa rớt) -> thay bằng ô 'mất kết nối' đã cache."""
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
# KHỞI TẠO CAMERA + CHƯƠNG TRÌNH CHÍNH
# ============================================================
def build_cameras():
    """Tạo danh sách Camera cho ô 1..N (chỉ số list = CAM (i+1)).
    - Có CAMERA_INSTANCE_IDS -> LUÔN dùng chế độ định danh theo instance_id (ổn
      định theo cổng USB). KHÔNG tự rơi về index thô kể cả khi thiếu lib, vì index
      thô bị XÁO khi rút/cắm lại (-> sai ô). Thiếu lib -> ô hiện 'mất kết nối' +
      cảnh báo to (phải bundle lib vào exe), KHÔNG xáo.
    - CAMERA_INSTANCE_IDS rỗng -> chế độ index thô CAMERA_INDICES (thủ công)."""
    n_cells = GRID_COLS * GRID_ROWS
    cameras = []
    if CAMERA_INSTANCE_IDS:
        cams = list_real_cameras()
        if cams is None:
            print("[!!] THIẾU thư viện cv2_enumerate_cameras -> KHÔNG resolve được "
                  "instance_id. Các ô sẽ hiện 'mất kết nối'. PHẢI đóng gói lib vào "
                  "exe (PyInstaller: --collect-all cv2_enumerate_cameras). "
                  "KHÔNG tự dùng index thô để tránh XÁO Ô khi cắm lại.")
        else:
            print("Ánh xạ CAM -> instance_id -> index:")
            preview = (list(CAMERA_INSTANCE_IDS) + [None] * n_cells)[:n_cells]
            for i, target in enumerate(preview):
                idx = find_index_by_instance(target, cams)
                tail = (target or "").split("\\")[-1]
                print(f"  CAM {i + 1}: ...{tail or '(trống)'} -> "
                      f"{'index ' + str(idx) if idx is not None else 'CHƯA THẤY'}")
        # Luôn dựng cam ở chế độ target (instance_id); _open() resolve mỗi lần mở.
        targets = (list(CAMERA_INSTANCE_IDS) + [None] * n_cells)[:n_cells]
        for i, target in enumerate(targets):
            cameras.append(Camera(name=f"CAM {i + 1}", number=i + 1,
                                  cell_size=(CELL_WIDTH, CELL_HEIGHT),
                                  target=target,
                                  rotate180=(i + 1) in ROTATE_180_CAMS))
        return cameras
    # CAMERA_INSTANCE_IDS rỗng -> index thô thủ công (chấp nhận có thể xáo khi cắm lại).
    srcs = (list(CAMERA_INDICES) + [None] * n_cells)[:n_cells]
    for i, src in enumerate(srcs):
        cameras.append(Camera(name=f"CAM {i + 1}", number=i + 1,
                              cell_size=(CELL_WIDTH, CELL_HEIGHT), src=src,
                              rotate180=(i + 1) in ROTATE_180_CAMS))
    return cameras


def _sequential_group_loop(cameras, slots, stop_event):
    """Luân phiên các cam trong 1 NHÓM TUẦN TỰ: mở 1 cam -> chụp ảnh vào ô -> giữ
    SEQ_DWELL giây -> ĐÓNG -> cam kế. LUÔN chỉ 1 cam trong nhóm MỞ cùng lúc (hợp
    giới hạn hub không cho 2 live). Chạy 1 thread nền; ô giữ ảnh gần nhất tới lượt sau."""
    members = [cameras[i] for i in slots if 0 <= i < len(cameras)]
    while not stop_event.is_set():
        for cam in members:
            if stop_event.is_set():
                break
            cam.snapshot()                 # mở (nếu cần) + chụp 1 ảnh tươi vào ô
            stop_event.wait(SEQ_DWELL)      # giữ ảnh một lúc (vẫn thoát nhanh)
            cam._release_cap()              # ĐÓNG để nhường cho cam kế trong nhóm
    for cam in members:
        cam._release_cap()


def main():
    print("Đang khởi tạo các camera...")
    cameras = build_cameras()

    # Slot nằm trong nhóm tuần tự (chụp luân phiên); còn lại chạy LIVE.
    seq_slots = set()
    for g in SEQUENTIAL_GROUPS:
        seq_slots.update(i for i in g if 0 <= i < len(cameras))

    # Cam LIVE: mỗi cam 1 thread đọc liên tục. Giãn start để USB negotiate ổn định.
    # Cam trong nhóm tuần tự KHÔNG start thread live (orchestrator điều khiển mở/đóng).
    for i, cam in enumerate(cameras):
        if i in seq_slots:
            continue
        cam.start()
        time.sleep(0.4)

    # Orchestrator cho từng nhóm tuần tự (1 thread/nhóm, chỉ 1 cam mở/lúc trong nhóm).
    stop_event = threading.Event()
    seq_threads = []
    for g in SEQUENTIAL_GROUPS:
        slots = [i for i in g if 0 <= i < len(cameras)]
        if not slots:
            continue
        t = threading.Thread(target=_sequential_group_loop,
                             args=(cameras, slots, stop_event), daemon=True)
        t.start()
        seq_threads.append(t)
    if seq_threads:
        groups_view = [[i + 1 for i in g] for g in SEQUENTIAL_GROUPS]
        print(f"Nhóm tuần tự (CAM, chỉ 1 cam mở/lúc trong nhóm): {groups_view}")

    # Pre-allocate canvas + cache ô "mất kết nối" (Pillow text rất chậm).
    canvas = allocate_canvas()
    disconnected_cells = build_disconnected_cells()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False
    print("Bắt đầu hiển thị. Phím: q/ESC thoát | f toàn màn hình | "
          "s chụp ảnh | R làm mới 6 cam")

    # Nhịp ghép lưới + hiển thị. Cap ~30fps để cửa sổ mượt + waitKey gọi đều
    # (Windows không báo "Not Responding").
    frame_period = 1.0 / 30.0
    next_tick = time.time()

    try:
        while True:
            # Watchdog "đứng hình" CHỈ cho cam LIVE: cam đang MỞ mà không có khung
            # mới quá FREEZE_TIMEOUT (grab() treo do rung USB) -> ép reset. Cam
            # trong nhóm tuần tự (cap tạm thời do orchestrator quản) -> bỏ qua.
            now = time.time()
            for i, cam in enumerate(cameras):
                if i in seq_slots:
                    continue
                if (cam.cap is not None and not cam._reset_requested
                        and cam.last_frame_time > 0.0
                        and now - cam.last_frame_time > FREEZE_TIMEOUT):
                    print(f"[watchdog] {cam.name} đứng hình "
                          f"{now - cam.last_frame_time:.1f}s -> ép reset.")
                    cam.request_reset()

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
            elif key in (ord("r"), ord("R")):
                # Làm mới TOÀN BỘ: cam LIVE -> request_reset (thread tự đóng & mở
                # lại, dò lại index, bỏ qua cooldown). Cam nhóm tuần tự -> chỉ đóng
                # cap, orchestrator sẽ mở lại ở lượt chụp kế.
                print("[R] Làm mới toàn bộ 6 camera (đóng & mở lại)...")
                for i, cam in enumerate(cameras):
                    if i in seq_slots:
                        cam._release_cap()
                    else:
                        cam.request_reset()

            # Giữ nhịp display: ngủ phần dư trong period.
            next_tick += frame_period
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()   # tụt nhịp -> reset, không tích nợ thời gian
    finally:
        print("Đang đóng camera...")
        stop_event.set()                       # dừng các orchestrator nhóm tuần tự
        for t in seq_threads:
            t.join(timeout=3.0)
        for i, cam in enumerate(cameras):
            if i in seq_slots:
                cam._release_cap()             # cam tuần tự: orchestrator đã dừng
            else:
                cam.stop()                     # cam live: dừng thread đọc + release
        cv2.destroyAllWindows()
        print("Đã thoát.")


def discover():
    """In bảng index | name | path của camera THẬT (đã lọc cam ảo) để lấy
    instance_id (nằm trong path) điền vào CAMERA_INSTANCE_IDS."""
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
    if "discover" in sys.argv:
        discover()
    elif "--detect" in sys.argv:
        detect_cameras()
    else:
        main()
