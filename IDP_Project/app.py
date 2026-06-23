import cv2
import numpy as np
import os
import json
import base64
import time
import threading
import re
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.models as models
import torchvision.transforms as T

# ── Vision Matrix Config ──────────────────────────────────────────────
BOX_X = 344; BOX_Y = 234; BOX_SCALE = 195
W_PCT = 0.94; H_PCT = 0.48; X_PCT = 0.5; Y_PCT = 0.48
ANALYSIS_THRESHOLD = 86
ORIENTATION_TOLERANCE = 2.0; PIXEL_SIZE_TOLERANCE = 47.0
DEEP_TEXTURE_THRESHOLD = 0.88; TOP_MATCH_MIN_SCORE = 0.91
STABLE_FRAME_COUNT = 10; SMOOTHING_ALPHA = 0.15; ANGLE_DEADBAND = 1.5

# ── STYLE CONSTANTS ───────────────────────────────────────────────────
BG_DARK      = "#1a1a2e"
BG_PANEL     = "#16213e"
BG_CARD      = "#0f3460"
BG_INPUT     = "#1a1a2e"
FG_WHITE     = "#e0e0e0"
FG_ACCENT    = "#00d2ff"
FG_GREEN     = "#00e676"
FG_RED       = "#ff5252"
FG_ORANGE    = "#ffd740"
FG_YELLOW    = "#ffff00"
FG_GREY      = "#9e9e9e"
BTN_PRIMARY  = "#0f3460"
BTN_SUCCESS  = "#00c853"
BTN_DANGER   = "#d50000"
BTN_WARNING  = "#ff6d00"
BTN_PURPLE   = "#7b1fa2"
BTN_DISABLED = "#37474f"

# ══════════════════════════════════════════════════════════════════════
#  HARDWARE CONTROLLER
# ══════════════════════════════════════════════════════════════════════
class HardwareController:
    def __init__(self, steps_per_rev=1600.0, cooldown_s=0.5):
        self.steps_per_rev = steps_per_rev
        self.degrees_per_step = 360.0 / self.steps_per_rev
        self.cooldown_s = cooldown_s
        self.last_transmission_time = 0.0
        self.ser = None
        self.auto_link_uart()

    def auto_link_uart(self):
        target_port = "COM17"
        try:
            print(f"[SERIAL] Connecting to {target_port}...")
            self.ser = serial.Serial(target_port, 9600, timeout=2.0)
            time.sleep(2.0)
            self.flush_buffers()
            self.ser.write(b"PING\n")
            response = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if response == "PONG":
                print(f"[SERIAL] Handshake OK on {target_port}.")
            else:
                print(f"[SERIAL] Unexpected response: '{response}'")
        except Exception as e:
            print(f"[SERIAL] Hardware simulation mode: {e}")

    def flush_buffers(self):
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

    def send_serial_string(self, p):
        if self.ser and self.ser.is_open:
            self.flush_buffers()
            self.ser.write(f"{p}\n".encode('utf-8'))
            self.ser.flush()
            print(f"[UART] Dispatched: {p}")
            self.last_transmission_time = time.time()

    def send_step_command(self, angle_deviation):
        if time.time() - self.last_transmission_time < self.cooldown_s:
            return
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"MOVE:0:0:{angle_deviation}\n".encode('utf-8'))
                self.ser.flush()
                print(f"[UART] Angle: {angle_deviation}°")
                self.last_transmission_time = time.time()
            except Exception as e:
                print(f"[UART] Error: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.write(b"EMERGENCY_STOP\n")
            self.ser.close()

# ══════════════════════════════════════════════════════════════════════
#  VISION ENGINE
# ══════════════════════════════════════════════════════════════════════
class IndustrialSlateEngine:
    def __init__(self, hardware_controller):
        self.hw = hardware_controller
        self.sm_angle = None
        self.target_angle_reference = None
        self.active_ui_tab = "INSPECTION"
        self.is_inspection_active = False
        self.stop_requested = False
        self.background_stack = []
        self.current_bg_capture_index = 0
        self.total_bg_slots_needed = 1
        self.is_holding_for_bg_sequence = False
        self.product_database = {}
        self.active_product_key = None
        self.has_trained_product = False
        self.trigger_record_product = False
        self.trigger_retake_bg = False
        self.is_currently_training = False
        self.training_progress_str = ""
        self.training_step_phase = "IDLE"
        self.background_memory = None
        self.last_trained_roi = None
        self.training_roi_front_collection = []
        self.training_roi_top_collection = []
        self.training_roi_back_collection = []
        self.training_roi_bottom_collection = []
        self.training_roi_bg_collection = []
        self.last_w = self.last_h = self.last_ang = 0
        self.stability_counter = 0
        self.is_profile_stable = False
        self.current_center_y = 240
        self.state_vote_buffer = []
        self.vote_buffer_max_size = 4
        self._last_size_window = None
        self.sharpen_kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_database_from_file()
        self.load_background_from_file()
        self.init_ai_models()

    def init_ai_models(self):
        print("[AI] Loading ResNet-18...")
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_roi_coordinates(self, box_x, box_y, box_scale, w_pct, h_pct, x_pct, y_pct):
        half = box_scale // 2
        bx1 = max(0, min(box_x - half, 640 - box_scale))
        by1 = max(0, min(box_y - half, 480 - box_scale))
        rect_w, rect_h = int(640 * w_pct), int(480 * h_pct)
        start_x = int(x_pct * 640) - (rect_w // 2)
        start_y = int(y_pct * 480) - (rect_h // 2)
        return bx1, by1, box_scale, max(0, start_x), max(0, start_y), min(640, start_x + rect_w), min(480, start_y + rect_h)

    def deskew_crop_roi(self, frame, cx, cy, w, h, angle):
        if frame is None or frame.size == 0:
            return np.zeros((48, 48, 3), dtype=np.uint8)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        return cv2.getRectSubPix(rotated, (max(1, int(w)), max(1, int(h))), (cx, cy))

    def calculate_deep_vector(self, roi_bgr):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        sharpened = cv2.filter2D(processed, -1, self.sharpen_kernel)
        with torch.no_grad():
            tensor = self.transform(sharpened).unsqueeze(0)
            vector = self.feature_extractor(tensor).flatten().numpy()
            return vector / (np.linalg.norm(vector) + 1e-6)

    def load_database_from_file(self):
        db_path = os.path.join(self.base_dir, "product_db.json")
        if os.path.exists(db_path):
            try:
                with open(db_path, "r") as f:
                    data = json.load(f)
                    for p_name, p_info in data.items():
                        p_info["fingerprints_front"] = [np.array(v) for v in p_info.get("fingerprints_front", [])]
                        p_info["fingerprints_top"] = [np.array(v) for v in p_info.get("fingerprints_top", [])]
                        p_info["fingerprints_back"] = [np.array(v) for v in p_info.get("fingerprints_back", [])]
                        p_info["fingerprints_bottom"] = [np.array(v) for v in p_info.get("fingerprints_bottom", [])]
                        if p_info.get("bg_bytes_base64"):
                            bg_bytes = base64.b64decode(p_info["bg_bytes_base64"])
                            p_info["bg_image_matrix"] = cv2.imdecode(np.frombuffer(bg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    self.product_database = data
                    if data:
                        self.active_product_key = list(data.keys())[0]
                        self.target_angle_reference = data[self.active_product_key]["angle_ref"]
                        self.has_trained_product = True
                        if data[self.active_product_key].get("bg_image_matrix") is not None:
                            self.background_memory = data[self.active_product_key]["bg_image_matrix"]
            except Exception as e:
                print(f"[DB ERROR] {e}")

    def load_background_from_file(self):
        bg_path = os.path.join(self.base_dir, "background_baseline.png")
        if self.background_memory is None and os.path.exists(bg_path):
            self.background_memory = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)

    def save_database_to_file(self):
        try:
            export_map = {}
            for p_name, p_info in self.product_database.items():
                bg_b64 = ""
                if p_info.get("bg_image_matrix") is not None:
                    _, buf = cv2.imencode('.png', p_info["bg_image_matrix"])
                    bg_b64 = base64.b64encode(buf).decode('utf-8')
                export_map[p_name] = {
                    "operator_id": p_info.get("operator_id", ""),
                    "angle_ref": p_info["angle_ref"],
                    "bg_bytes_base64": bg_b64,
                    "fingerprints_front": [v.tolist() for v in p_info.get("fingerprints_front", [])],
                    "fingerprints_top": [v.tolist() for v in p_info.get("fingerprints_top", [])],
                    "fingerprints_back": [v.tolist() for v in p_info.get("fingerprints_back", [])],
                    "fingerprints_bottom": [v.tolist() for v in p_info.get("fingerprints_bottom", [])]
                }
            with open(os.path.join(self.base_dir, "product_db.json"), "w") as f:
                json.dump(export_map, f, indent=4)
        except Exception as e:
            print(f"[DISK ERROR] {e}")

    def execute_manual_front_capture(self, cap, w, h, geo, cx, cy):
        self.is_currently_training = True
        self.temp_f_front = []; self.training_roi_front_collection = []
        captured = 0
        while captured < 25:
            ret, frame = cap.read()
            if not ret: continue
            bx1, by1, bs, x1, y1, x2, y2 = self.get_roi_coordinates(*geo)
            pf = cv2.resize(frame[by1:by1+bs, bx1:bx1+bs], (640, 480))
            roi = self.deskew_crop_roi(pf, cx, cy, w, h, self.sm_angle)
            if roi.size > 0:
                self.temp_f_front.append(self.calculate_deep_vector(roi))
                self.training_roi_front_collection.append(roi); captured += 1
                self.training_progress_str = f"FRONT ({captured}/25)..."
                time.sleep(0.002)
        self.training_step_phase = "WAIT_TOP"; self.is_currently_training = False

    def execute_manual_top_capture(self, cap, w, h, geo, cx, cy):
        self.is_currently_training = True
        self.temp_f_top = []; self.training_roi_top_collection = []
        captured = 0
        while captured < 25:
            ret, frame = cap.read()
            if not ret: continue
            bx1, by1, bs, x1, y1, x2, y2 = self.get_roi_coordinates(*geo)
            pf = cv2.resize(frame[by1:by1+bs, bx1:bx1+bs], (640, 480))
            roi = self.deskew_crop_roi(pf, cx, cy, w, h, self.sm_angle)
            if roi.size > 0:
                self.temp_f_top.append(self.calculate_deep_vector(roi))
                self.training_roi_top_collection.append(roi); captured += 1
                self.training_progress_str = f"TOP ({captured}/25)..."
                time.sleep(0.002)
        self.training_step_phase = "WAIT_BACK"; self.is_currently_training = False

    def execute_manual_back_capture(self, cap, w, h, geo, cx, cy):
        self.is_currently_training = True
        self.temp_f_back = []; self.training_roi_back_collection = []
        captured = 0
        while captured < 25:
            ret, frame = cap.read()
            if not ret: continue
            bx1, by1, bs, x1, y1, x2, y2 = self.get_roi_coordinates(*geo)
            pf = cv2.resize(frame[by1:by1+bs, bx1:bx1+bs], (640, 480))
            roi = self.deskew_crop_roi(pf, cx, cy, w, h, self.sm_angle)
            if roi.size > 0:
                self.temp_f_back.append(self.calculate_deep_vector(roi))
                self.training_roi_back_collection.append(roi); captured += 1
                self.training_progress_str = f"BACK ({captured}/25)..."
                time.sleep(0.002)
        self.training_step_phase = "WAIT_BOTTOM"; self.is_currently_training = False

    def execute_manual_bottom_capture(self, cap, prod_name, op_id, w, h, geo, cx, cy, cb):
        self.is_currently_training = True
        self.temp_f_bottom = []; self.training_roi_bottom_collection = []
        captured = 0
        while captured < 25:
            ret, frame = cap.read()
            if not ret: continue
            bx1, by1, bs, x1, y1, x2, y2 = self.get_roi_coordinates(*geo)
            pf = cv2.resize(frame[by1:by1+bs, bx1:bx1+bs], (640, 480))
            roi = self.deskew_crop_roi(pf, cx, cy, w, h, self.sm_angle)
            if roi.size > 0:
                self.temp_f_bottom.append(self.calculate_deep_vector(roi))
                self.training_roi_bottom_collection.append(roi); captured += 1
                self.training_progress_str = f"BOTTOM ({captured}/25)..."
                time.sleep(0.002)
        self.product_database[prod_name] = {
            "operator_id": op_id, "angle_ref": self.sm_angle,
            "bg_image_matrix": self.background_memory,
            "fingerprints_front": self.temp_f_front, "fingerprints_top": self.temp_f_top,
            "fingerprints_back": self.temp_f_back, "fingerprints_bottom": self.temp_f_bottom
        }
        self.active_product_key = prod_name
        self.target_angle_reference = self.sm_angle
        self.training_step_phase = "IDLE"; self.is_currently_training = False
        cb()

    def identify_product_match(self, roi, cw, ch, sz_tol, tex_thresh, min_score=0.91):
        if roi is None or roi.size == 0:
            return "WRONG_PRODUCT", 0.0, ""
        if not self.active_product_key or self.active_product_key not in self.product_database:
            return "DATABASE_EMPTY", 0.0, ""
        m = self.product_database[self.active_product_key]
        r180 = cv2.rotate(roi, cv2.ROTATE_180)
        v0 = self.calculate_deep_vector(roi); v180 = self.calculate_deep_vector(r180)
        sc = {
            "MATCH_BOTTOM": max([np.dot(v0, t) for t in m.get("fingerprints_front", [])] or [0]),
            "FLIP_180": max([np.dot(v180, t) for t in m.get("fingerprints_front", [])] or [0]),
            "MATCH_TOP": max([np.dot(v0, t) for t in m.get("fingerprints_top", [])] or [0]),
            "TOP_180_FLIP": max([np.dot(v180, t) for t in m.get("fingerprints_top", [])] or [0]),
            "REVERSE_SIDE_FRONT": max([np.dot(v0, t) for t in m.get("fingerprints_back", [])] or [0]),
            "BOTTOM_SIDE": max([np.dot(v0, t) for t in m.get("fingerprints_bottom", [])] or [0]),
            "BOTTOM_180_FLIP": max([np.dot(v180, t) for t in m.get("fingerprints_bottom", [])] or [0]),
        }
        best = max(sc, key=sc.get); best_score = sc[best]
        if best_score < tex_thresh:
            return "WRONG_PRODUCT", best_score, ""
        self.state_vote_buffer.append(best)
        if len(self.state_vote_buffer) > self.vote_buffer_max_size:
            self.state_vote_buffer.pop(0)
        states, counts = np.unique(self.state_vote_buffer, return_counts=True)
        return states[np.argmax(counts)], best_score, self.active_product_key

# ══════════════════════════════════════════════════════════════════════
#  HMI APPLICATION
# ══════════════════════════════════════════════════════════════════════
class BenchApp(tk.Tk):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.title("VISION CONTROL — HMI BENCH STATION")
        self.geometry("1280x720")
        self.minsize(1024, 600)
        self.configure(bg=BG_DARK)

        self.cap = None
        self.parameter_values = {}
        self.param_labels = {}
        self.is_running = True
        self.calculated_clamp_ms = None
        self.thumbnail_images_cache = []
        self.is_conveyor_running = False
        self.training_button_state = "TRAIN_FRONT"
        self.diagnostic_step_index = 1

        self._apply_global_styles()
        self.load_runtime_defaults()
        self.build_ui_layout()
        self.update_dropdown_lists()
        self.load_presets_index()
        self.restore_last_used_preset()

        self.notebook.select(self.tab_inspect)
        self.engine.active_ui_tab = "INSPECTION"
        self.after(100, self.evaluate_live_inspection_orientation_gate)

    def _apply_global_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=BG_DARK, foreground=FG_WHITE, fieldbackground=BG_INPUT)
        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=FG_WHITE,
                        padding=[14, 6], font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", BTN_PRIMARY)])
        style.configure("TFrame", background=BG_DARK)
        style.configure("TLabelframe", background=BG_PANEL, foreground=FG_ACCENT,
                        font=("Segoe UI", 9, "bold"))
        style.configure("TLabelframe.Label", background=BG_PANEL, foreground=FG_ACCENT)
        style.configure("TCombobox", fieldbackground=BG_INPUT, foreground=FG_WHITE,
                        arrowcolor=FG_WHITE, selectbackground=BTN_PRIMARY)
        style.map("TCombobox", fieldbackground=[("readonly", BG_INPUT)])
        style.configure("TSeparator", background=BG_CARD)

    # ── BUILD UI ──────────────────────────────────────────────────────
    def build_ui_layout(self):
        left_panel = tk.Frame(self, bg=BG_PANEL, width=540)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        left_panel.pack_propagate(False)

        header = tk.Label(left_panel, text="◆ VISION CONTROL STATION",
                          bg=BG_PANEL, fg=FG_ACCENT,
                          font=("Segoe UI", 12, "bold"))
        header.pack(fill=tk.X, padx=12, pady=(10, 4))

        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.tab_train = tk.Frame(self.notebook, bg=BG_PANEL)
        self.tab_inspect = tk.Frame(self.notebook, bg=BG_PANEL)
        self.tab_calibrate = tk.Frame(self.notebook, bg=BG_PANEL)
        self.tab_actuators = tk.Frame(self.notebook, bg=BG_PANEL)

        self.notebook.add(self.tab_train, text="1. TRAINING")
        self.notebook.add(self.tab_inspect, text="2. INSPECTION")
        self.notebook.add(self.tab_calibrate, text="3. TUNING")
        self.notebook.add(self.tab_actuators, text="4. ACTUATORS")

        self.setup_training_tab()
        self.setup_inspection_tab()
        self.setup_tuning_tab()
        self.setup_actuators_tab()

        right_panel = tk.Frame(self, bg=BG_DARK)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=8)

        self.lbl_status = tk.Label(right_panel, text="CAMERA FEED OFFLINE.",
                                   bg=BG_DARK, fg=FG_GREY,
                                   font=("Segoe UI", 11, "bold"))
        self.lbl_status.pack(anchor=tk.W, padx=10, pady=(6, 4))

        self.video_canvas = tk.Label(right_panel, bg="#000000", relief=tk.SUNKEN, bd=2)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    # ── PRODUCT SERIES SELECTION ──────────────────────────────────────
    def select_product_series_macro(self, series_id):
        for i in range(1, 4):
            getattr(self, f"btn_series_{i}").configure(bg=BG_PANEL, fg=FG_WHITE)
        target_btn = getattr(self, f"btn_series_{series_id}")
        if self.selected_series == series_id:
            self.series_click_count += 1
        else:
            self.selected_series = series_id
            self.series_click_count = 1
        if self.series_click_count == 1:
            target_btn.configure(bg=BTN_PRIMARY, fg=FG_ACCENT)
            self.lbl_series_status.configure(
                text=f"Active: Product Series {series_id} (Registered)",
                fg=FG_ACCENT)
        elif self.series_click_count >= 2:
            target_btn.configure(bg=BTN_SUCCESS, fg="#ffffff")
            self.lbl_series_status.configure(
                text=f"Active: Product Series {series_id} (Confirmed) → Zone 2 & 3",
                fg=FG_GREEN)
            if self.engine.hw.ser and self.engine.hw.ser.is_open:
                cmd = {1: "TRIGGER_PROD_1", 2: "TRIGGER_PROD_2", 3: "TRIGGER_WRONG"}
                self.engine.hw.send_serial_string(cmd[series_id])

    # ── LOAD RUNTIME DEFAULTS ─────────────────────────────────────────
    def load_runtime_defaults(self):
        self.parameter_values = {
            cv2.CAP_PROP_EXPOSURE: -4 + 12, cv2.CAP_PROP_ZOOM: 100,
            cv2.CAP_PROP_FOCUS: 0, cv2.CAP_PROP_BRIGHTNESS: 109,
            cv2.CAP_PROP_CONTRAST: 97, cv2.CAP_PROP_SHARPNESS: 255,
            cv2.CAP_PROP_SATURATION: 255, cv2.CAP_PROP_WB_TEMPERATURE: 4500,
            "BOX_X": BOX_X, "BOX_Y": BOX_Y, "BOX_SCALE": BOX_SCALE,
            "W_PCT": int(W_PCT * 100), "H_PCT": int(H_PCT * 100),
            "X_PCT": int(X_PCT * 100), "Y_PCT": int(Y_PCT * 100),
            "THRESHOLD": ANALYSIS_THRESHOLD, "ORIENT": int(ORIENTATION_TOLERANCE),
            "SIZE": int(PIXEL_SIZE_TOLERANCE),
            "TEXTURE": int(DEEP_TEXTURE_THRESHOLD * 100),
            "TOP_MIN_SCORE": TOP_MATCH_MIN_SCORE,
            "STABLE_FRAMES": STABLE_FRAME_COUNT,
            "LINE1_Y": 100, "LINE2_Y": 380
        }

    # ── TRAINING TAB ──────────────────────────────────────────────────
    def setup_training_tab(self):
        c = self.tab_train
        sf = tk.Frame(c, bg=BG_CARD, bd=1, relief=tk.RIDGE)
        sf.pack(fill=tk.X, padx=10, pady=6)
        self.lbl_hw_status = tk.Label(sf, text="HARDWARE: CHECKING...",
                                      bg=BG_CARD, fg=FG_ORANGE,
                                      font=("Segoe UI", 9, "bold"))
        self.lbl_hw_status.pack(pady=4)

        tk.Label(c, text="[STEP 1] TRACEABILITY ENTRY",
                 bg=BG_PANEL, fg=FG_ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=12, pady=(8, 2))
        tk.Label(c, text="Product Name:", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Segoe UI", 8)).pack(anchor=tk.W, padx=12, pady=1)
        self.txt_prod_name = tk.Entry(c, bg=BG_INPUT, fg=FG_WHITE,
                                      insertbackground=FG_WHITE, relief=tk.FLAT,
                                      font=("Segoe UI", 9))
        self.txt_prod_name.pack(fill=tk.X, padx=12, pady=2)
        self.txt_prod_name.bind("<KeyRelease>", self.validate_training_workflow_state)

        ttk.Separator(c, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        tk.Label(c, text="Background Samples:", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Segoe UI", 8)).pack(anchor=tk.W, padx=12, pady=1)
        self.txt_bg_target = tk.Entry(c, width=6, bg=BG_INPUT, fg=FG_WHITE,
                                      insertbackground=FG_WHITE, relief=tk.FLAT,
                                      font=("Segoe UI", 9))
        self.txt_bg_target.insert(0, "1")
        self.txt_bg_target.pack(anchor=tk.W, padx=12, pady=2)

        self.btn_init_bg = self._styled_btn(c, "LOCK BACKGROUND", BTN_DISABLED,
                                            FG_GREY, tk.DISABLED)
        self.btn_init_bg.pack(fill=tk.X, padx=12, pady=2)

        self.btn_reuse_bg = self._styled_btn(c, "REUSE BACKGROUND MEMORY",
                                             BTN_DISABLED, FG_GREY, tk.DISABLED,
                                             self.bypass_and_reuse_background)
        self.btn_reuse_bg.pack(fill=tk.X, padx=12, pady=3)

        self.lbl_bg_prompt = tk.Label(c, text="Baseline: Locked. Complete Step 1.",
                                      bg=BG_PANEL, fg=FG_GREY,
                                      font=("Segoe UI", 8, "italic"))
        self.lbl_bg_prompt.pack(anchor=tk.W, padx=12, pady=2)

        ttk.Separator(c, orient="horizontal").pack(fill=tk.X, padx=8, pady=8)

        self.btn_train_product = self._styled_btn(c, "TRAIN FRONT SIDE",
                                                  BTN_DISABLED, FG_GREY, tk.DISABLED,
                                                  self.handle_training_button_click)
        self.btn_train_product.pack(fill=tk.X, padx=12, pady=3)

        self.btn_direct_reset = self._styled_btn(c, "↺ RESET / RETAKE FACE",
                                                 BTN_WARNING, "#ffffff", tk.NORMAL,
                                                 self.abort_training_pipeline_manually)
        self.btn_direct_reset.pack(fill=tk.X, padx=12, pady=3)

        bf = tk.Frame(c, bg=BG_PANEL)
        bf.pack(fill=tk.X, padx=12, pady=4)
        self.btn_test_sequence = self._styled_btn(bf, "⚙ OVERALL TEST",
                                                  BTN_PURPLE, "#ffffff", tk.NORMAL,
                                                  self.run_automated_mechatronics_flip_sequence)
        self.btn_test_sequence.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_step_sequence = self._styled_btn(bf, "▶ NEXT STEP",
                                                  BTN_PRIMARY, "#ffffff", tk.NORMAL,
                                                  self.trigger_discrete_step_test)
        self.btn_step_sequence.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=1)

        self.lbl_step_status = tk.Label(bf, text="Step 1/5 (Ready to Clamp)",
                                        bg=BG_PANEL, fg=FG_GREY,
                                        font=("Segoe UI", 8, "italic"))
        self.lbl_step_status.pack(pady=2)

        self.btn_wipe_db = self._styled_btn(c, "⚠ CLEAR LOCAL DATABASE",
                                            BTN_DANGER, "#ffffff", tk.NORMAL,
                                            self.wipe_product_database_handler)
        self.btn_wipe_db.pack(fill=tk.X, padx=12, pady=12)

    # ── INSPECTION TAB ────────────────────────────────────────────────
    def setup_inspection_tab(self):
        c = self.tab_inspect

        tk.Label(c, text="PRODUCT SERIES", bg=BG_PANEL, fg=FG_ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
        sbf = tk.Frame(c, bg=BG_PANEL)
        sbf.pack(fill="x", padx=12, pady=4)
        self.btn_series_1 = self._styled_btn(sbf, "SERIES 1", BG_PANEL, FG_WHITE,
                                             tk.NORMAL, lambda: self.select_product_series_macro(1))
        self.btn_series_1.pack(side="left", expand=True, fill="x", padx=1)
        self.btn_series_2 = self._styled_btn(sbf, "SERIES 2", BG_PANEL, FG_WHITE,
                                             tk.NORMAL, lambda: self.select_product_series_macro(2))
        self.btn_series_2.pack(side="left", expand=True, fill="x", padx=1)
        self.btn_series_3 = self._styled_btn(sbf, "SERIES 3", BG_PANEL, FG_WHITE,
                                             tk.NORMAL, lambda: self.select_product_series_macro(3))
        self.btn_series_3.pack(side="left", expand=True, fill="x", padx=1)
        self.lbl_series_status = tk.Label(c, text="Active: None",
                                          bg=BG_PANEL, fg=FG_ORANGE,
                                          font=("Segoe UI", 8, "italic"))
        self.lbl_series_status.pack(anchor="w", padx=12, pady=(0, 8))

        ttk.Separator(c, orient="horizontal").pack(fill=tk.X, padx=8, pady=4)

        tk.Label(c, text="AUTOMATION INSPECTION ENGINE",
                 bg=BG_PANEL, fg=f"#{'%02x%02x%02x' % tuple(int(FG_ACCENT[i:i+2], 16) for i in (1,3,5))}",
                 font=("Segoe UI", 10, "bold")).pack(pady=8)

        self.cmb_inspect_products = ttk.Combobox(c, state="readonly")
        self.cmb_inspect_products.pack(fill=tk.X, padx=20, pady=4)
        self.cmb_inspect_products.bind("<<ComboboxSelected>>", self.on_dropdown_model_changed)

        self._styled_btn(c, "▶ START INSPECTION", BTN_SUCCESS, "#ffffff",
                         tk.NORMAL, self.start_inspection_routine).pack(fill=tk.X, padx=20, pady=4)

        self.btn_fix_orientation = self._styled_btn(c, "🔧 FIX ORIENTATION",
                                                    BTN_DISABLED, FG_GREY, tk.DISABLED,
                                                    self.trigger_live_orientation_realign_macro)
        self.btn_fix_orientation.pack(fill=tk.X, padx=20, pady=4)

        self.btn_proceed_zones = self._styled_btn(c, "⏩ PROCEED ZONE 2 & 3",
                                                  BTN_DISABLED, FG_GREY, tk.DISABLED,
                                                  self.trigger_zone_2_3_proceed_macro)
        self.btn_proceed_zones.pack(fill=tk.X, padx=20, pady=4)

        self._styled_btn(c, "■ STOP INSPECTION", BTN_DANGER, "#ffffff",
                         tk.NORMAL, self.stop_inspection_routine).pack(fill=tk.X, padx=20, pady=4)

        self.lbl_inspect_details = tk.Label(c, text="Pipeline: CAMERA OFF\nSelect profile and Start.",
                                            bg=BG_PANEL, fg=FG_GREY, justify=tk.LEFT,
                                            font=("Segoe UI", 8))
        self.lbl_inspect_details.pack(anchor=tk.W, padx=20, pady=10)

    # ── ACTUATORS TAB ─────────────────────────────────────────────────
    def setup_actuators_tab(self):
        c = self.tab_actuators
        canvas = tk.Canvas(c, bg=BG_PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(c, orient="vertical", command=canvas.yview)
        sf = tk.Frame(canvas, bg=BG_PANEL)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        sb.pack(side="right", fill="y")

        def dispatch(payload):
            if self.engine.hw.ser and self.engine.hw.ser.is_open:
                self.engine.hw.ser.reset_input_buffer()
                self.engine.hw.ser.write(f"{payload}\n".encode('utf-8'))
                print(f"[SERIAL] {payload}")
            else:
                messagebox.showwarning("UART Offline", "Arduino disconnected.")

        def run_calc():
            if not self.engine.is_profile_stable or self.engine.last_w == 0:
                self.lbl_clamp_calc_status.configure(text="Error: No stable object", fg=FG_RED)
                self.btn_execute_intel_clamp.configure(state=tk.DISABLED, bg=BTN_DISABLED)
                self.btn_execute_intel_unclamp.configure(state=tk.DISABLED, bg=BTN_DISABLED)
                return
            scale = float(self.parameter_values["BOX_SCALE"])
            ratio = 180.0 / scale
            w_mm = self.engine.last_w * ratio
            if w_mm <= 36.5:
                self.lbl_clamp_calc_status.configure(text=f"Width {w_mm:.1f}mm ≤ 36.5mm", fg=FG_YELLOW)
                return
            disp = w_mm - 36.5
            calc = int(disp / (19.5 / 800.0))
            self.calculated_clamp_ms = max(100, min(calc + 50, 2500))
            self.lbl_clamp_calc_status.configure(text=f"Window: {self.calculated_clamp_ms} ms", fg=FG_GREEN)
            self.btn_execute_intel_clamp.configure(state=tk.NORMAL, bg=BTN_SUCCESS)
            self.btn_execute_intel_unclamp.configure(state=tk.NORMAL, bg=BTN_PRIMARY)

        def compile_flip(mid):
            d1 = ent_d_cw.get().strip() if mid == 1 else ent_d_ccw.get().strip()
            d2 = ent_e_cw.get().strip() if mid == 1 else ent_e_ccw.get().strip()
            dispatch(f"SYSTEM_TWIN_FLIP:{d1}:{d2}:{mid}")

        # Module 1 – Spin
        f1 = ttk.LabelFrame(sf, text=" CONTINUOUS SPIN (CH 3 & 4) ")
        f1.pack(fill="x", padx=8, pady=4)
        self._styled_btn(f1, "CW SPIN", BTN_DANGER, "#ffffff", tk.NORMAL,
                         lambda: dispatch("MANUAL_SPIN:1:1")).grid(row=0, column=0, padx=2, pady=4)
        self._styled_btn(f1, "CCW SPIN", "#00838f", "#ffffff", tk.NORMAL,
                         lambda: dispatch("MANUAL_SPIN:2:1")).grid(row=0, column=1, padx=2, pady=4)
        self._styled_btn(f1, "■ STOP", BG_DARK, FG_ORANGE, tk.NORMAL,
                         lambda: dispatch("MANUAL_SPIN:0:0")).grid(row=0, column=2, padx=2, pady=4)

        # Module 2 – Clamp
        f2 = ttk.LabelFrame(sf, text=" CLAMP JAW (CH 0) ")
        f2.pack(fill="x", padx=8, pady=4)
        self._styled_btn(f2, "1. RUN CALCULATION", "#00838f", "#ffffff",
                         tk.NORMAL, run_calc).grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.lbl_clamp_calc_status = tk.Label(f2, text="Awaiting stable target...",
                                              bg=BG_PANEL, fg=FG_ORANGE,
                                              font=("Segoe UI", 8, "italic"))
        self.lbl_clamp_calc_status.grid(row=0, column=1, columnspan=2, padx=8, pady=6, sticky="w")
        self.btn_execute_intel_clamp = self._styled_btn(f2, "2. INTEL-CLAMP",
                                                        BTN_DISABLED, FG_GREY, tk.DISABLED,
                                                        lambda: dispatch(f"DRIVE:0:{self.calculated_clamp_ms}:1"))
        self.btn_execute_intel_clamp.grid(row=1, column=0, padx=4, pady=4, sticky="ew")
        self.btn_execute_intel_unclamp = self._styled_btn(f2, "3. RETRACT (+150ms)",
                                                          BTN_DISABLED, FG_GREY, tk.DISABLED,
                                                          lambda: dispatch(f"DRIVE:0:{self.calculated_clamp_ms + 150}:2"))
        self.btn_execute_intel_unclamp.grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        # Module 3 – Hoist
        f3 = ttk.LabelFrame(sf, text=" HOIST LIFTERS (CH 1 & CH 2) ")
        f3.pack(fill="x", padx=8, pady=4)
        tk.Label(f3, text="Duration (ms):", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Segoe UI", 8)).grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ent_h = tk.Entry(f3, width=6, bg=BG_INPUT, fg=FG_GREEN, insertbackground=FG_WHITE,
                         font=("Consolas", 10, "bold"), relief=tk.FLAT)
        ent_h.insert(0, "600"); ent_h.grid(row=0, column=1, padx=4, pady=4)
        self._styled_btn(f3, "LIFT UP", "#0d47a1", "#ffffff", tk.NORMAL,
                         lambda: dispatch(f"COMBINED_HOIST:{ent_h.get().strip()}:1")).grid(row=0, column=2, padx=4)
        self._styled_btn(f3, "LIFT DOWN", BG_PANEL, "#ffffff", tk.NORMAL,
                         lambda: dispatch(f"COMBINED_HOIST:{ent_h.get().strip()}:2")).grid(row=0, column=3, padx=4)

        # Module 4 – Twin Flips
        f4 = ttk.LabelFrame(sf, text=" TWIN FLIPS (CH 3 & 4) ")
        f4.pack(fill="x", padx=8, pady=4)
        fd = ttk.LabelFrame(f4, text=" Flip D (CH 3) ")
        fd.pack(fill="x", padx=6, pady=2)
        tk.Label(fd, text="CW (ms):", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Segoe UI", 8)).grid(row=0, column=0, padx=4)
        ent_d_cw = tk.Entry(fd, width=5, bg=BG_INPUT, fg=FG_GREEN,
                            insertbackground=FG_WHITE, relief=tk.FLAT)
        ent_d_cw.insert(0, "600"); ent_d_cw.grid(row=0, column=1, padx=4)
        tk.Label(fd, text="CCW (ms):", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Segoe UI", 8)).grid(row=1, column=0, padx=4)
        ent_d_ccw = tk.Entry(fd, width=5, bg=BG_INPUT, fg=FG_GREEN,
                             insertbackground=FG_WHITE, relief=tk.FLAT)
        ent_d_ccw.insert(0, "650"); ent_d_ccw.grid(row=1, column=1, padx=4)

        fe = ttk.LabelFrame(f4, text=" Flip E (CH 4) ")
        fe.pack(fill="x", padx=6, pady=2)
        tk.Label(fe, text="CW (ms):", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Segoe UI", 8)).grid(row=0, column=0, padx=4)
        ent_e_cw = tk.Entry(fe, width=5, bg=BG_INPUT, fg=FG_GREEN,
                            insertbackground=FG_WHITE, relief=tk.FLAT)
        ent_e_cw.insert(0, "600"); ent_e_cw.grid(row=0, column=1, padx=4)
        tk.Label(fe, text="CCW (ms):", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Segoe UI", 8)).grid(row=1, column=0, padx=4)
        ent_e_ccw = tk.Entry(fe, width=5, bg=BG_INPUT, fg=FG_GREEN,
                             insertbackground=FG_WHITE, relief=tk.FLAT)
        ent_e_ccw.insert(0, "700"); ent_e_ccw.grid(row=1, column=1, padx=4)

        self._styled_btn(f4, "FLIP 90° CW", BTN_DANGER, "#ffffff", tk.NORMAL,
                         lambda: compile_flip(1)).pack(fill="x", padx=10, pady=4)
        self._styled_btn(f4, "FLIP 90° CCW", "#00838f", "#ffffff", tk.NORMAL,
                         lambda: compile_flip(2)).pack(fill="x", padx=10, pady=4)

        self._styled_btn(sf, "■ EMERGENCY STOP", BTN_DANGER, "#ffffff",
                         tk.NORMAL, lambda: dispatch("EMERGENCY_STOP")).pack(fill=tk.X, padx=8, pady=12)

    # ── TUNING TAB ────────────────────────────────────────────────────
    def setup_tuning_tab(self):
        c = self.tab_calibrate
        canvas = tk.Canvas(c, bg=BG_PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(c, orient="vertical", command=canvas.yview)
        sf = tk.Frame(canvas, bg=BG_PANEL)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=4, pady=4)
        sb.pack(side="right", fill="y")

        def adj(key, step, mx, mn=0):
            v = self.parameter_values[key] + step
            self.parameter_values[key] = max(mn, min(v, mx))
            self.param_labels[key].configure(text=str(self.parameter_values[key]))

        def build_row(label, key_id, mx, mn=0):
            f = tk.Frame(sf, bg=BG_PANEL)
            f.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(f, text=label, bg=BG_PANEL, fg=FG_WHITE, width=22,
                     anchor=tk.W, font=("Segoe UI", 8)).pack(side=tk.LEFT)
            steps = [-100, -50, -10] if key_id == cv2.CAP_PROP_WB_TEMPERATURE else [-10, -5, -1]
            psteps = [10, 50, 100] if key_id == cv2.CAP_PROP_WB_TEMPERATURE else [1, 5, 10]
            for s in steps:
                tk.Button(f, text=str(s), width=3, bg=BG_CARD, fg=FG_WHITE, bd=0,
                          font=("Segoe UI", 7, "bold"),
                          command=lambda x=s: adj(key_id, x, mx, mn)).pack(side=tk.LEFT, padx=1)
            lbl = tk.Label(f, text=str(self.parameter_values[key_id]),
                           bg=BG_INPUT, fg=FG_GREEN, width=5,
                           font=("Consolas", 10, "bold"), relief=tk.FLAT)
            lbl.pack(side=tk.LEFT, padx=4)
            self.param_labels[key_id] = lbl
            for s in psteps:
                tk.Button(f, text=f"+{s}", width=3, bg=BG_CARD, fg=FG_WHITE, bd=0,
                          font=("Segoe UI", 7, "bold"),
                          command=lambda x=s: adj(key_id, x, mx, mn)).pack(side=tk.LEFT, padx=1)

        def adj_float(key, step, mx, mn=0.0):
            v = round(self.parameter_values[key] + step, 4)
            self.parameter_values[key] = max(mn, min(v, mx))
            self.param_labels[key].configure(text=f"{self.parameter_values[key]:.2f}")

        def build_row_float(label, key_id, mx, mn=0.0):
            f = tk.Frame(sf, bg=BG_PANEL)
            f.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(f, text=label, bg=BG_PANEL, fg=FG_WHITE, width=22,
                     anchor=tk.W, font=("Segoe UI", 8)).pack(side=tk.LEFT)
            for s in [-0.1, -0.05, -0.01]:
                tk.Button(f, text=str(s), width=3, bg=BG_CARD, fg=FG_WHITE, bd=0,
                          font=("Segoe UI", 7, "bold"),
                          command=lambda x=s: adj_float(key_id, x, mx, mn)).pack(side=tk.LEFT, padx=1)
            lbl = tk.Label(f, text=f"{self.parameter_values[key_id]:.2f}",
                           bg=BG_INPUT, fg=FG_GREEN, width=5,
                           font=("Consolas", 10, "bold"), relief=tk.FLAT)
            lbl.pack(side=tk.LEFT, padx=4)
            self.param_labels[key_id] = lbl
            for s in [0.01, 0.05, 0.1]:
                tk.Button(f, text=f"+{s}", width=3, bg=BG_CARD, fg=FG_WHITE, bd=0,
                          font=("Segoe UI", 7, "bold"),
                          command=lambda x=s: adj_float(key_id, x, mx, mn)).pack(side=tk.LEFT, padx=1)

        tk.Label(sf, text="PRESETS", bg=BG_PANEL, fg=FG_ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=10, pady=6)
        pf = tk.Frame(sf, bg=BG_PANEL)
        pf.pack(fill=tk.X, padx=10, pady=4)
        self.txt_preset_name = tk.Entry(pf, width=12, bg=BG_INPUT, fg=FG_WHITE,
                                        insertbackground=FG_WHITE, relief=tk.FLAT)
        self.txt_preset_name.insert(0, "Default_Line")
        self.txt_preset_name.pack(side=tk.LEFT, padx=2)
        self.cmb_presets = ttk.Combobox(pf, width=14, state="readonly")
        self.cmb_presets.pack(side=tk.LEFT, padx=2)
        self.cmb_presets.bind("<<ComboboxSelected>>", lambda e: (
            self.txt_preset_name.delete(0, tk.END),
            self.txt_preset_name.insert(0, e.widget.get())))
        self._styled_btn(pf, "LOAD", BTN_PRIMARY, "#ffffff", tk.NORMAL,
                         self.load_preset_to_ui).pack(side=tk.LEFT, padx=1)
        self._styled_btn(pf, "SAVE", BTN_SUCCESS, "#ffffff", tk.NORMAL,
                         self.save_preset_to_json).pack(side=tk.LEFT, padx=1)

        tk.Label(sf, text="CAMERA", bg=BG_PANEL, fg=FG_ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=10, pady=6)
        build_row("Exposure Offset:", cv2.CAP_PROP_EXPOSURE, 12, 0)
        build_row("Zoom:", cv2.CAP_PROP_ZOOM, 500, 100)
        build_row("Focus:", cv2.CAP_PROP_FOCUS, 255, 0)
        build_row("Brightness:", cv2.CAP_PROP_BRIGHTNESS, 255, 0)
        build_row("Contrast:", cv2.CAP_PROP_CONTRAST, 255, 0)
        build_row("White Balance:", cv2.CAP_PROP_WB_TEMPERATURE, 6500, 2000)

        tk.Label(sf, text="GEOMETRY", bg=BG_PANEL, fg=FG_ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=10, pady=6)
        build_row("Box Center X:", "BOX_X", 640, 0)
        build_row("Box Center Y:", "BOX_Y", 480, 0)
        build_row("Box Scale:", "BOX_SCALE", 640, 100)
        build_row("ROI Width %:", "W_PCT", 100, 0)
        build_row("ROI Height %:", "H_PCT", 100, 0)

        tk.Label(sf, text="FILTERS", bg=BG_PANEL, fg=FG_ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=10, pady=6)
        build_row("Threshold:", "THRESHOLD", 255, 0)
        build_row("Orientation Tol:", "ORIENT", 45, 0)
        build_row("Size Tol:", "SIZE", 200, 0)
        build_row("Texture Match %:", "TEXTURE", 100, 50)
        build_row_float("Min Score:", "TOP_MIN_SCORE", 1.0, 0.0)
        build_row("Stable Frames:", "STABLE_FRAMES", 100, 1)
        build_row("Line 1 Y:", "LINE1_Y", 480, 0)
        build_row("Line 2 Y:", "LINE2_Y", 480, 0)

        self._styled_btn(sf, "SAVE CONFIG TO SOURCE", BTN_SUCCESS, "#ffffff",
                         tk.NORMAL, self.trigger_disk_save).pack(fill=tk.X, padx=10, pady=16)

    # ── HELPERS ───────────────────────────────────────────────────────
    def _styled_btn(self, parent, text, bg, fg, state, cmd=None):
        btn = tk.Button(parent, text=text, bg=bg, fg=fg, state=state,
                        font=("Segoe UI", 8, "bold"), bd=0, padx=8, pady=6,
                        activebackground=bg, activeforeground=fg,
                        cursor="hand2", command=cmd)
        btn.bind("<Enter>", lambda e: btn.configure(bg="#ffffff", fg=BG_DARK if bg != BTN_DANGER else BTN_DANGER) if state==tk.NORMAL else None)
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg, fg=fg))
        return btn

    def validate_training_workflow_state(self, event=None):
        name = self.txt_prod_name.get().strip()
        if name and self.engine.training_step_phase == "IDLE" and not self.engine.is_holding_for_bg_sequence:
            self.btn_init_bg.configure(state=tk.NORMAL, bg=BTN_PRIMARY, fg="#ffffff")
            self.btn_reuse_bg.configure(state=tk.NORMAL, bg="#5d4037", fg="#ffffff")
        elif self.engine.training_step_phase == "IDLE":
            self.btn_init_bg.configure(state=tk.DISABLED, text="LOCK BACKGROUND", bg=BTN_DISABLED, fg=FG_GREY)
            self.btn_reuse_bg.configure(state=tk.DISABLED, bg=BTN_DISABLED, fg=FG_GREY)
            self.btn_train_product.configure(state=tk.DISABLED, text="TRAIN FRONT SIDE", bg=BTN_DISABLED, fg=FG_GREY)

    def handle_training_button_click(self):
        s = self.training_button_state
        if s == "TRAIN_FRONT":
            self.engine.trigger_record_product = True
            self.check_training_completion_loop("WAIT_TOP", "EXECUTE_FLIP_TOP", "FLIP TO TOP POSITION")
        elif s == "EXECUTE_FLIP_TOP":
            self.btn_train_product.configure(state=tk.DISABLED, text="ROTATING...")
            threading.Thread(target=self.execute_discrete_training_flip,
                             args=(650, 2, "TRAIN_TOP", "CAPTURE TOP VIEW"), daemon=True).start()
        elif s == "TRAIN_TOP":
            self.engine.training_step_phase = "WAIT_TOP"
            self.engine.trigger_record_product = True
            self.check_training_completion_loop("WAIT_BACK", "EXECUTE_FLIP_BACK", "FLIP TO BACK POSITION")
        elif s == "EXECUTE_FLIP_BACK":
            self.btn_train_product.configure(state=tk.DISABLED, text="ROTATING...")
            threading.Thread(target=self.execute_discrete_training_flip,
                             args=(650, 2, "TRAIN_BACK", "CAPTURE BACK VIEW"), daemon=True).start()
        elif s == "TRAIN_BACK":
            self.engine.training_step_phase = "WAIT_BACK"
            self.engine.trigger_record_product = True
            self.check_training_completion_loop("WAIT_BOTTOM", "EXECUTE_FLIP_BOTTOM", "FLIP TO BOTTOM POSITION")
        elif s == "EXECUTE_FLIP_BOTTOM":
            self.btn_train_product.configure(state=tk.DISABLED, text="ROTATING...")
            threading.Thread(target=self.execute_discrete_training_flip,
                             args=(650, 2, "TRAIN_BOTTOM", "CAPTURE BOTTOM VIEW"), daemon=True).start()
        elif s == "TRAIN_BOTTOM":
            self.engine.training_step_phase = "WAIT_BOTTOM"
            self.engine.trigger_record_product = True

    def execute_discrete_training_flip(self, dur, direc, ns, bl):
        try:
            scale = float(self.parameter_values["BOX_SCALE"])
            w_mm = self.engine.last_w * (180.0 / scale)
            disp = w_mm - 36.5
            calc = int(disp / (19.5 / 800.0))
            cm = max(100, min(calc + 50, 2500))
            self.send_serial_string(f"DRIVE:0:{cm}:1")
            time.sleep((cm / 1000.0) + 0.4)
            self.send_serial_string("COMBINED_HOIST:800:1")
            time.sleep(1.2)
            self.send_serial_string(f"SYSTEM_TWIN_FLIP:{dur}:{dur}:{direc}")
            time.sleep((dur / 1000.0) + 0.4)
            self.send_serial_string("COMBINED_HOIST:800:2")
            time.sleep(1.2)
            self.send_serial_string(f"DRIVE:0:{cm + 150}:2")
            time.sleep(((cm + 150) / 1000.0) + 0.5)
            self.after(0, lambda: [
                self.btn_train_product.configure(state=tk.NORMAL, text=bl, bg=BTN_SUCCESS),
                setattr(self, 'training_button_state', ns)])
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Flip Error", str(e)))

    def check_training_completion_loop(self, target, nxt, txt):
        if self.engine.training_step_phase == target:
            self.training_button_state = nxt
            self.btn_train_product.configure(state=tk.NORMAL, text=txt, bg=BTN_WARNING)
        else:
            self.after(100, lambda: self.check_training_completion_loop(target, nxt, txt))

    def trigger_discrete_step_test(self):
        self.btn_step_sequence.configure(state=tk.DISABLED, text="DRIVING...")
        threading.Thread(target=self.run_single_discrete_diagnostic_step, daemon=True).start()

    def run_single_discrete_diagnostic_step(self):
        try:
            scale = float(self.parameter_values["BOX_SCALE"])
            w_mm = self.engine.last_w * (180.0 / scale)
            disp = w_mm - 36.5
            calc = int(disp / (19.5 / 800.0))
            cm = max(100, min(calc + 50, 2500))
            if self.diagnostic_step_index == 1:
                self.send_serial_string(f"DRIVE:0:{cm}:1")
                time.sleep((cm / 1000.0) + 0.2)
                self.diagnostic_step_index = 2
                self.after(0, lambda: self.lbl_step_status.configure(text="Step 2/5: Lift Up", fg=FG_ORANGE))
            elif self.diagnostic_step_index == 2:
                self.send_serial_string("COMBINED_HOIST:800:1")
                time.sleep(1.0)
                self.diagnostic_step_index = 3
                self.after(0, lambda: self.lbl_step_status.configure(text="Step 3/5: Flip 90°", fg=FG_ORANGE))
            elif self.diagnostic_step_index == 3:
                self.send_serial_string("SYSTEM_TWIN_FLIP:600:600:1")
                time.sleep(0.8)
                self.diagnostic_step_index = 4
                self.after(0, lambda: self.lbl_step_status.configure(text="Step 4/5: Lower Down", fg=FG_ORANGE))
            elif self.diagnostic_step_index == 4:
                self.send_serial_string("COMBINED_HOIST:800:2")
                time.sleep(1.0)
                self.diagnostic_step_index = 5
                self.after(0, lambda: self.lbl_step_status.configure(text="Step 5/5: Unclamp", fg=FG_ORANGE))
            elif self.diagnostic_step_index == 5:
                self.send_serial_string(f"DRIVE:0:{cm + 150}:2")
                time.sleep(((cm + 150) / 1000.0) + 0.2)
                self.diagnostic_step_index = 1
                self.after(0, lambda: self.lbl_step_status.configure(text="Step 1/5: Ready to Clamp", fg=FG_GREY))
            self.after(0, lambda: self.btn_step_sequence.configure(state=tk.NORMAL, text="▶ NEXT STEP"))
        except Exception as e:
            self.after(0, lambda: [messagebox.showerror("Step Error", str(e)),
                                   self.btn_step_sequence.configure(state=tk.NORMAL, text="▶ NEXT STEP")])

    def run_automated_mechatronics_flip_sequence(self):
        try:
            scale = float(self.parameter_values["BOX_SCALE"])
            w_mm = self.engine.last_w * (180.0 / scale)
            disp = w_mm - 36.5
            calc = int(disp / (19.5 / 800.0))
            cm = max(100, min(calc + 50, 2500))
            self.send_serial_string(f"DRIVE:0:{cm}:1")
            time.sleep((cm / 1000.0) + 0.4)
            self.send_serial_string("COMBINED_HOIST:800:1")
            time.sleep(1.2)
            self.send_serial_string("SYSTEM_TWIN_FLIP:650:700:2")
            time.sleep(1.1)
            self.send_serial_string("COMBINED_HOIST:800:2")
            time.sleep(1.2)
            self.send_serial_string(f"DRIVE:0:{cm + 150}:2")
            time.sleep(((cm + 150) / 1000.0) + 0.5)
            self.after(0, self.transition_to_top_view_capture_ready)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Automation Error", str(e)))

    def evaluate_live_inspection_orientation_gate(self):
        try:
            if not self.engine.is_inspection_active:
                self.btn_fix_orientation.configure(state=tk.DISABLED, bg=BTN_DISABLED, fg=FG_GREY, text="🔧 FIX ORIENTATION")
                self.btn_proceed_zones.configure(state=tk.DISABLED, bg=BTN_DISABLED, fg=FG_GREY)
                self.after(200, self.evaluate_live_inspection_orientation_gate)
                return
            cv = str(self.engine.state_vote_buffer[-1]) if self.engine.state_vote_buffer else "UNKNOWN"
            if cv in ["REVERSE_SIDE_FRONT", "MATCH_TOP", "TOP_180_FLIP", "BOTTOM_SIDE", "BOTTOM_180_FLIP", "FLIP_180"]:
                self.btn_fix_orientation.configure(state=tk.NORMAL, bg=BTN_WARNING, fg="#ffffff", text="🔧 FIX ORIENTATION")
            else:
                self.btn_fix_orientation.configure(state=tk.DISABLED, bg=BTN_DISABLED, fg=FG_GREY, text="🔧 FIX ORIENTATION")
            if cv in ["MATCH_BOTTOM", "FLIP_180", "REVERSE_SIDE_FRONT"]:
                self.btn_proceed_zones.configure(state=tk.NORMAL, bg=BTN_PRIMARY, fg="#ffffff")
            else:
                self.btn_proceed_zones.configure(state=tk.DISABLED, bg=BTN_DISABLED, fg=FG_GREY)
        except Exception:
            pass
        self.after(200, self.evaluate_live_inspection_orientation_gate)

    def trigger_live_orientation_realign_macro(self):
        self.btn_fix_orientation.configure(state=tk.DISABLED, text="DRIVING GANTRY...")
        threading.Thread(target=self.process_live_mechatronics_realign_thread, daemon=True).start()

    def process_live_mechatronics_realign_thread(self):
        try:
            scale = float(self.parameter_values["BOX_SCALE"])
            w_mm = self.engine.last_w * (180.0 / scale)
            disp = w_mm - 36.5
            calc = int(disp / (19.5 / 800.0))
            cm = max(100, min(calc + 50, 2500))
            self.send_serial_string(f"DRIVE:0:{cm}:1")
            time.sleep((cm / 1000.0) + 0.4)
            self.send_serial_string("COMBINED_HOIST:800:1")
            time.sleep(1.2)
            cv = str(self.engine.state_vote_buffer[-1]) if self.engine.state_vote_buffer else "UNKNOWN"
            d1 = abs(self.engine.current_center_y - int(self.parameter_values["LINE1_Y"]))
            d2 = abs(self.engine.current_center_y - int(self.parameter_values["LINE2_Y"]))
            near_l1 = d1 < d2
            if near_l1:
                mapping = {
                    "REVERSE_SIDE_FRONT": ("SYSTEM_TWIN_FLIP:1200:1200:1", 1.6),
                    "BACKSIDE": ("SYSTEM_TWIN_FLIP:1200:1200:1", 1.6),
                    "MATCH_TOP": ("SYSTEM_TWIN_FLIP:600:600:1", 1.0),
                    "TOP_180_FLIP": ("SYSTEM_TWIN_FLIP:1800:1800:1", 2.2),
                    "BOTTOM_SIDE": ("SYSTEM_TWIN_FLIP:1800:1800:1", 2.2),
                    "BOTTOM_180_FLIP": ("SYSTEM_TWIN_FLIP:600:600:1", 1.0)
                }
            else:
                mapping = {
                    "REVERSE_SIDE_FRONT": ("SYSTEM_TWIN_FLIP:1300:1350:2", 1.75),
                    "BACKSIDE": ("SYSTEM_TWIN_FLIP:1300:1350:2", 1.75),
                    "MATCH_TOP": ("SYSTEM_TWIN_FLIP:600:600:1", 1.0),
                    "TOP_180_FLIP": ("SYSTEM_TWIN_FLIP:650:700:2", 1.1),
                    "BOTTOM_SIDE": ("SYSTEM_TWIN_FLIP:650:700:2", 1.1),
                    "BOTTOM_180_FLIP": ("SYSTEM_TWIN_FLIP:600:600:1", 1.0)
                }
            cmd, slp = mapping.get(cv, ("SYSTEM_TWIN_FLIP:600:600:1", 1.0))
            self.send_serial_string(cmd)
            time.sleep(slp)
            self.send_serial_string("COMBINED_HOIST:800:2")
            time.sleep(1.2)
            self.send_serial_string(f"DRIVE:0:{cm + 150}:2")
            time.sleep(((cm + 150) / 1000.0) + 0.5)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Corrector Error", str(e)))
        finally:
            self.after(0, lambda: self.btn_fix_orientation.configure(state=tk.NORMAL, text="🔧 FIX ORIENTATION", bg=BTN_WARNING))

    def trigger_zone_2_3_proceed_macro(self):
        detected = self.engine.active_product_key
        if detected and "brown" in detected.lower():
            print(f"[HMI] Routing {detected} → Y1")
            self.engine.hw.send_serial_string("TRIGGER_PROD_2")
        elif detected and any(c in detected.lower() for c in ["green", "yellow", "orange"]):
            print(f"[HMI] Routing {detected} → Y0")
            self.engine.hw.send_serial_string("TRIGGER_PROD_1")
        else:
            print("[HMI] Unrecognized → Scrap")
            self.engine.hw.send_serial_string("TRIGGER_WRONG")

    def abort_training_pipeline_manually(self):
        self.engine.training_step_phase = "IDLE"
        self.engine.trigger_record_product = False
        self.engine.is_currently_training = False
        self.engine.training_roi_front_collection.clear()
        self.engine.training_roi_top_collection.clear()
        self.engine.training_roi_back_collection.clear()
        self.engine.training_roi_bottom_collection.clear()
        self.engine.training_roi_bg_collection.clear()
        self.training_button_state = "TRAIN_FRONT"
        self.btn_train_product.configure(text="TRAIN FRONT SIDE", bg=BTN_PRIMARY)
        self.txt_prod_name.delete(0, tk.END)
        self.validate_training_workflow_state()
        messagebox.showinfo("Reset", "Training buffers cleared.")

    def on_training_complete(self):
        w = tk.Toplevel(self)
        w.title("Profile Verification")
        w.configure(bg=BG_DARK)
        w.grab_set(); w.state('zoomed')
        gf = tk.Frame(w, bg=BG_DARK)
        gf.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        self.thumbnail_images_cache = []

        def build_row(p, r, title, coll):
            f = tk.LabelFrame(p, text=f" {title.upper()} ", bg=BG_PANEL, fg=FG_ACCENT,
                              font=("Segoe UI", 9, "bold"))
            f.grid(row=r, column=0, sticky="nsew", pady=3, padx=4)
            cv = tk.Canvas(f, bg=BG_INPUT, highlightthickness=0, height=80)
            sb = ttk.Scrollbar(f, orient="horizontal", command=cv.xview)
            ig = tk.Frame(cv, bg=BG_INPUT)
            cv.create_window((0, 0), window=ig, anchor="nw")
            cv.configure(xscrollcommand=sb.set)
            ig.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
            cv.pack(side="top", fill="both", expand=True)
            sb.pack(side="bottom", fill="x")
            if title.lower() == "background" and self.engine.background_memory is not None:
                bg_res = cv2.resize(self.engine.background_memory, (90, 70))
                t = ImageTk.PhotoImage(image=Image.fromarray(bg_res))
                self.thumbnail_images_cache.append(t)
                tk.Label(ig, image=t, bg="#000000", bd=1).pack(side=tk.LEFT, padx=4, pady=4)
            else:
                for roi in coll:
                    if roi is not None and roi.size > 0:
                        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        t = ImageTk.PhotoImage(image=Image.fromarray(rgb).resize((90, 70)))
                        self.thumbnail_images_cache.append(t)
                        tk.Label(ig, image=t, bg="#000000", bd=1).pack(side=tk.LEFT, padx=4, pady=4)

        gf.grid_columnconfigure(0, weight=1)
        for i in range(5): gf.grid_rowconfigure(i, weight=1)
        build_row(gf, 0, "background", [])
        build_row(gf, 1, "front", getattr(self.engine, 'training_roi_front_collection', []))
        build_row(gf, 2, "top", getattr(self.engine, 'training_roi_top_collection', []))
        build_row(gf, 3, "back", getattr(self.engine, 'training_roi_back_collection', []))
        build_row(gf, 4, "bottom", getattr(self.engine, 'training_roi_bottom_collection', []))

        cf = tk.Frame(w, bg=BG_PANEL, bd=1, relief=tk.RAISED)
        cf.pack(fill=tk.X, pady=6, padx=16)
        tk.Label(cf, text="Operator:", bg=BG_PANEL, fg=FG_WHITE,
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=12, pady=2)
        tx = tk.Entry(cf, bg=BG_INPUT, fg=FG_WHITE, insertbackground=FG_WHITE,
                      font=("Consolas", 11), relief=tk.FLAT)
        tx.pack(fill=tk.X, padx=12, pady=4)
        tx.insert(0, "OP_LINE_01")

        def save():
            o = tx.get().strip()
            if not o: return
            n = self.engine.active_product_key
            if n in self.engine.product_database:
                self.engine.product_database[n]["operator_id"] = o
            self.engine.save_database_to_file()
            self.engine.has_trained_product = True
            self.update_dropdown_lists()
            self.txt_prod_name.delete(0, tk.END)
            self.validate_training_workflow_state()
            w.destroy(); self.notebook.select(self.tab_inspect)

        def discard():
            if self.engine.active_product_key in self.engine.product_database:
                del self.engine.product_database[self.engine.active_product_key]
            self.engine.active_product_key = None
            self.engine.training_roi_front_collection.clear()
            self.engine.training_roi_top_collection.clear()
            self.engine.training_roi_back_collection.clear()
            self.engine.training_roi_bottom_collection.clear()
            self.engine.training_roi_bg_collection.clear()
            self.training_button_state = "TRAIN_FRONT"
            self.btn_train_product.configure(text="TRAIN FRONT SIDE", bg=BTN_PRIMARY)
            self.validate_training_workflow_state()
            w.destroy()

        bf = tk.Frame(w, bg=BG_DARK, pady=8)
        bf.pack(fill=tk.X)
        self._styled_btn(bf, "■ ABORT & RESET", BTN_DANGER, "#ffffff",
                         tk.NORMAL, discard).pack(side=tk.LEFT, padx=20)
        self._styled_btn(bf, "▶ COMMIT SAMPLES", BTN_SUCCESS, "#ffffff",
                         tk.NORMAL, save).pack(side=tk.RIGHT, padx=20)

    def transition_to_top_view_capture_ready(self):
        self.training_button_state = "TRAIN_TOP"
        self.btn_train_product.configure(state=tk.NORMAL, text="TAKE TOP VIEW", bg=BTN_SUCCESS)

    def on_dropdown_model_changed(self, e):
        self.engine.active_product_key = e.widget.get()
        self.update_dropdown_lists()

    def update_dropdown_lists(self):
        m = list(self.engine.product_database.keys())
        self.cmb_inspect_products.configure(values=m)
        if self.engine.active_product_key:
            self.cmb_inspect_products.set(self.engine.active_product_key)

    def bypass_and_reuse_background(self):
        if self.engine.background_memory is not None:
            self.engine.training_step_phase = "WAIT_FRONT"
            self.btn_train_product.configure(state=tk.NORMAL, text="TRAIN FRONT SIDE", bg=BTN_PRIMARY)

    def wipe_product_database_handler(self):
        if messagebox.askyesno("Wipe Database", "Delete all trained items?"):
            self.engine.product_database = {}
            self.update_dropdown_lists()

    def load_presets_index(self):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment_presets.json")
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    d = json.load(f)
                    self.cmb_presets.configure(values=list(d.keys()))
                    if d: self.cmb_presets.set(list(d.keys())[0])
            except Exception: pass

    def save_preset_to_json(self):
        name = self.txt_preset_name.get().strip()
        if not name: return
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment_presets.json")
        presets = {}
        if os.path.exists(p):
            try:
                with open(p, "r") as f: presets = json.load(f)
            except Exception: pass
        presets[name] = {str(k): v for k, v in self.parameter_values.items()}
        with open(p, "w") as f: json.dump(presets, f, indent=4)
        self.record_last_used_preset_token(name)
        self.load_presets_index()
        messagebox.showinfo("Success", f"Preset '{name}' saved.")

    def load_preset_to_ui(self):
        name = self.cmb_presets.get()
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "environment_presets.json")
        if not name or not os.path.exists(p): return
        try:
            with open(p, "r") as f: presets = json.load(f)
            if name in presets:
                for k, v in presets[name].items():
                    tk_key = int(k) if k.isdigit() or (k.startswith('-') and k[1:].isdigit()) else k
                    if tk_key in self.parameter_values: self.parameter_values[tk_key] = v
                self.record_last_used_preset_token(name)
                for key_id, lbl in self.param_labels.items():
                    if key_id in self.parameter_values:
                        t = f"{self.parameter_values[key_id]:.2f}" if isinstance(self.parameter_values[key_id], float) else str(self.parameter_values[key_id])
                        lbl.configure(text=t)
            messagebox.showinfo("Success", f"Preset '{name}' loaded.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def record_last_used_preset_token(self, name):
        try:
            with open(os.path.join(self.engine.base_dir, "last_active_preset.txt"), "w") as f:
                f.write(name)
        except Exception: pass

    def restore_last_used_preset(self):
        sp = os.path.join(self.engine.base_dir, "last_active_preset.txt")
        pp = os.path.join(self.engine.base_dir, "environment_presets.json")
        if os.path.exists(sp) and os.path.exists(pp):
            try:
                with open(sp, "r") as f: name = f.read().strip()
                with open(pp, "r") as f: presets = json.load(f)
                if name in presets:
                    self.cmb_presets.set(name)
                    self.txt_preset_name.delete(0, tk.END)
                    self.txt_preset_name.insert(0, name)
                    for k, v in presets[name].items():
                        tk_key = int(k) if k.isdigit() or (k.startswith('-') and k[1:].isdigit()) else k
                        if tk_key in self.parameter_values: self.parameter_values[tk_key] = v
            except Exception: pass

    def trigger_disk_save(self):
        try:
            base = os.path.dirname(os.path.abspath(__file__))
            main_path = os.path.join(base, "app.py")
            with open(main_path, "r", encoding="utf-8") as f: code = f.read()
            for key, var in [("BOX_X", "BOX_X"), ("BOX_Y", "BOX_Y"), ("BOX_SCALE", "BOX_SCALE"),
                             ("ANALYSIS_THRESHOLD", "THRESHOLD")]:
                code = re.sub(rf"{key} = \d+", f"{key} = {self.parameter_values[var]}", code)
            with open(main_path, "w", encoding="utf-8") as f: f.write(code)
            messagebox.showinfo("Success", "Config saved to source.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def start_inspection_routine(self):
        self.engine.is_inspection_active = True

    def stop_inspection_routine(self):
        self.engine.is_inspection_active = False
        self.render_image_on_canvas(np.zeros((480, 640, 3), dtype=np.uint8))

    def render_image_on_canvas(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_canvas.configure(image=img)
        self.video_canvas.image = img

    def send_serial_string(self, p):
        if self.engine.hw.ser and self.engine.hw.ser.is_open:
            self.engine.hw.ser.write(f"{p}\n".encode())
            print(f"[UART] {p}")

# ══════════════════════════════════════════════════════════════════════
#  VIDEO PIPELINE
# ══════════════════════════════════════════════════════════════════════
def video_pipeline_worker(app, engine):
    time.sleep(0.5)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    last_frame_time = time.time()
    last_applied = {}
    force_first = True

    while app.is_running:
        now = time.time()
        if now - last_frame_time < 0.033:
            time.sleep(0.033 - (now - last_frame_time))
            continue
        last_frame_time = now
        if engine.stop_requested:
            time.sleep(0.05); continue
        try:
            raw = app.notebook.tab(app.notebook.select(), "text")
            tn = raw.split(".")[-1].strip().upper()
            engine.active_ui_tab = tn
        except Exception:
            tn = "TRAINING"
        need_cam = (tn in ["TRAINING","TUNING","ACTUATORS"]) or (tn=="INSPECTION" and engine.is_inspection_active)
        if not need_cam:
            if app.cap:
                app.cap.release(); app.cap = None; force_first = True
            if tn == "INSPECTION":
                app.render_image_on_canvas(np.zeros((480,640,3),dtype=np.uint8))
            time.sleep(0.05); continue
        if app.cap is None or not app.cap.isOpened():
            if engine.active_ui_tab in ["TRAINING","TUNING","ACTUATORS"] or engine.is_inspection_active:
                if app.cap:
                    try: app.cap.release()
                    except: pass
                app.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                time.sleep(0.2)
                if not app.cap.isOpened():
                    app.cap = cv2.VideoCapture(0)
                app.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                app.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                force_first = True
            else:
                time.sleep(0.05); continue

        pv = app.parameter_values
        lbx,lby,lbs = pv["BOX_X"],pv["BOX_Y"],pv["BOX_SCALE"]
        lwp, lhp = pv["W_PCT"]/100.0, pv["H_PCT"]/100.0
        lxp, lyp = pv["X_PCT"]/100.0, pv["Y_PCT"]/100.0
        th = pv["THRESHOLD"]; ot = float(pv["ORIENT"]); st = float(pv["SIZE"])
        tl = pv["TEXTURE"]/100.0; ms = float(pv["TOP_MIN_SCORE"])
        tw = pv["STABLE_FRAMES"]; ly1 = int(pv["LINE1_Y"]); ly2 = int(pv["LINE2_Y"])

        if app.cap and app.cap.isOpened():
            props = [cv2.CAP_PROP_EXPOSURE,cv2.CAP_PROP_ZOOM,cv2.CAP_PROP_FOCUS,
                     cv2.CAP_PROP_BRIGHTNESS,cv2.CAP_PROP_CONTRAST,cv2.CAP_PROP_WB_TEMPERATURE]
            changed = any(pv[p] != last_applied.get(p) for p in props)
            if force_first or changed:
                app.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1)
                app.cap.set(cv2.CAP_PROP_AUTOFOCUS,0)
                app.cap.set(cv2.CAP_PROP_AUTO_WB,0)
                app.cap.set(cv2.CAP_PROP_EXPOSURE, pv[cv2.CAP_PROP_EXPOSURE]-12)
                app.cap.set(cv2.CAP_PROP_FOCUS, pv[cv2.CAP_PROP_FOCUS])
                app.cap.set(cv2.CAP_PROP_ZOOM, pv[cv2.CAP_PROP_ZOOM])
                app.cap.set(cv2.CAP_PROP_BRIGHTNESS, pv[cv2.CAP_PROP_BRIGHTNESS])
                app.cap.set(cv2.CAP_PROP_CONTRAST, pv[cv2.CAP_PROP_CONTRAST])
                app.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, pv[cv2.CAP_PROP_WB_TEMPERATURE])
                for p in props: last_applied[p] = pv[p]
                force_first = False

        ret, frame = (app.cap.read() if app.cap else (False, None))
        if not ret or frame is None: time.sleep(0.01); continue

        gp = (lbx,lby,lbs,lwp,lhp,lxp,lyp)
        bx1,by1,bs,x1,y1,x2,y2 = engine.get_roi_coordinates(*gp)
        df = cv2.resize(frame[by1:by1+bs,bx1:bx1+bs], (640,480))
        mf = np.zeros_like(df); mf[y1:y2,x1:x2] = df[y1:y2,x1:x2]
        mf = cv2.filter2D(mf, -1, engine.sharpen_kernel)

        if hasattr(app, 'lbl_hw_status'):
            if engine.hw.ser and engine.hw.ser.is_open:
                app.lbl_hw_status.configure(text="✔ ARDUINO MEGA: ONLINE (COM17)", fg=FG_GREEN)
            else:
                app.lbl_hw_status.configure(text="❌ ARDUINO MEGA: OFFLINE", fg=FG_RED)

        gray = cv2.GaussianBlur(cv2.cvtColor(mf,cv2.COLOR_BGR2GRAY),(5,5),0)
        delta = cv2.absdiff(engine.background_memory,gray) if engine.background_memory is not None else gray
        _, tm = cv2.threshold(delta, th, 255, cv2.THRESH_BINARY)
        am = np.zeros_like(tm); am[y1:y2,x1:x2] = tm[y1:y2,x1:x2]
        cnts,_ = cv2.findContours(am, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None; max_a = 0
        for c in cnts:
            a = cv2.contourArea(c)
            if a > max_a and a > 100: max_a = a; best = c

        ss = f"MODE: {tn} | MONITORING..."
        hc = (255,255,255)

        if best is not None:
            rect = cv2.minAreaRect(best); (cx,cy),(wb,hb),ra = rect
            if wb < hb: wb,hb = hb,wb; ra += 90.0
            if ra > 90.0: ra -= 180.0
            elif ra < -90.0: ra += 180.0
            if abs(wb-engine.last_w)<1.5 and abs(hb-engine.last_h)<1.5 and abs(ra-engine.last_ang)<1.5:
                engine.stability_counter += 1
            else:
                engine.stability_counter = 0; engine.is_profile_stable = False
            engine.last_w,engine.last_h,engine.last_ang = wb,hb,ra
            if engine.stability_counter >= tw: engine.is_profile_stable = True

            if engine.is_profile_stable:
                if engine.sm_angle is None: engine.sm_angle = ra
                else:
                    ad = ra - engine.sm_angle
                    if ad > 180: ad -= 360
                    elif ad < -180: ad += 360
                    if abs(ad) > ANGLE_DEADBAND: engine.sm_angle += SMOOTHING_ALPHA * ad
                engine.current_center_y = int(cy)
                if tn == "TRAINING":
                    ss = f"TRAINING | Phase: {engine.training_step_phase} | {int(wb)}x{int(hb)}px"
                    hc = (0,255,0)
                else:
                    roi = engine.deskew_crop_roi(mf,cx,cy,wb,hb,engine.sm_angle)
                    ms_,mv,mn = engine.identify_product_match(roi,wb,hb,st,tl,ms)
                    spin = round(engine.sm_angle,1)
                    app.evaluate_live_inspection_orientation_gate()
                    TABLE = {
                        "DATABASE_EMPTY": (f"NO PROFILE ({int(wb)}x{int(hb)}px)", (255,255,255)),
                        "WRONG_PRODUCT": ("DIFFERENT SERIES DETECTED", (19,69,139)),
                        "MATCH_BOTTOM": (f"MATCH: '{engine.active_product_key}' | FRONT FACE OK", (0,255,0)),
                        "FLIP_180": (f"FLIP 180 | Shift: {spin}°", (0,165,255)),
                        "REVERSE_SIDE_FRONT": (f"BACKSIDE DETECTED | Shift: {spin}°", (255,191,0)),
                        "MATCH_TOP": ("TOP FACE ALIGNED", (255,255,255)),
                        "TOP_180_FLIP": ("TOP 180 FLIP", (0,0,0)),
                        "BOTTOM_SIDE": ("BOTTOM SIDE", (142,36,170)),
                        "BOTTOM_180_FLIP": ("BOTTOM 180 FLIP", (128,128,128))
                    }
                    s2, h2 = TABLE.get(ms_, ("UNKNOWN", (255,255,255)))
                    if ms_ == "MATCH_BOTTOM" and abs(spin) > ot:
                        s2, h2 = f"ORIENTATION MISFIT | Shift: {spin}°", (0,255,255)
                    ss, hc = s2, h2
                    if tn == "INSPECTION" and engine.is_inspection_active:
                        app.send_serial_string(f"MOVE:0:{spin}")
            else:
                ss = f"STABILIZING ({engine.stability_counter}/{tw}) | {int(wb)}x{int(hb)}px"
                hc = (0,165,255)
            tra = engine.sm_angle if engine.sm_angle is not None else ra
            cv2.drawContours(df,[cv2.boxPoints(((cx,cy),(wb,hb),tra)).astype(np.int32)],0,hc,2)
            cv2.circle(df,(int(cx),int(cy)),4,(0,255,255),-1)
        else:
            if not (tn=="TRAINING" and engine.is_holding_for_bg_sequence):
                engine.sm_angle = engine.stability_counter = 0; engine.is_profile_stable = False
                engine.sm_angle = None
                ss = f"MODE: {engine.active_ui_tab} | IDLE"; hc = (0,255,0)

        if tn=="TRAINING" and engine.is_holding_for_bg_sequence:
            cv2.rectangle(df,(x1,y1),(x2,y2),(255,255,255),1)
            bf = cv2.GaussianBlur(cv2.cvtColor(mf,cv2.COLOR_BGR2GRAY),(5,5),0)
            if engine.trigger_retake_bg:
                engine.trigger_retake_bg = False
                engine.background_stack.append(bf)
                engine.current_bg_capture_index += 1
                if engine.current_bg_capture_index >= engine.total_bg_slots_needed:
                    engine.background_memory = np.mean(engine.background_stack,axis=0).astype(np.uint8)
                    cv2.imwrite(os.path.join(base_dir,"background_baseline.png"),engine.background_memory)
                    engine.is_holding_for_bg_sequence = False
                    app.btn_init_bg.configure(text="LOCK BACKGROUND", bg=BTN_PRIMARY)
                    app.lbl_bg_prompt.configure(text=f"✔ Baselines Saved ({len(engine.background_stack)} slots)",fg=FG_GREEN)
                    engine.training_step_phase = "WAIT_FRONT"
                    app.btn_train_product.configure(state=tk.NORMAL,text="TRAIN FRONT SIDE",bg=BTN_SUCCESS,fg="#ffffff")
                else:
                    idx = engine.current_bg_capture_index
                    tot = engine.total_bg_slots_needed
                    app.btn_init_bg.configure(text=f"CAPTURE ({idx+1}/{tot})")
                    app.lbl_bg_prompt.configure(text=f"→ Modify tray for position {idx+1}",fg=FG_ORANGE)
            ss = "BG SEQUENCE"; hc = (0,255,255)

        if engine.trigger_record_product and engine.is_profile_stable and 'best' in locals() and best is not None:
            engine.trigger_record_product = False
            if engine.training_step_phase == "WAIT_FRONT":
                app.btn_train_product.configure(text="RECORDING FRONT...", state=tk.DISABLED)
                threading.Thread(target=engine.execute_manual_front_capture, args=(app.cap,wb,hb,gp,cx,cy),daemon=True).start()
            elif engine.training_step_phase == "WAIT_TOP":
                app.btn_train_product.configure(text="RECORDING TOP...", state=tk.DISABLED)
                threading.Thread(target=engine.execute_manual_top_capture, args=(app.cap,wb,hb,gp,cx,cy),daemon=True).start()
            elif engine.training_step_phase == "WAIT_BACK":
                app.btn_train_product.configure(text="RECORDING BACK...", state=tk.DISABLED)
                threading.Thread(target=engine.execute_manual_back_capture, args=(app.cap,wb,hb,gp,cx,cy),daemon=True).start()
            elif engine.training_step_phase == "WAIT_BOTTOM":
                app.btn_train_product.configure(text="RECORDING BOTTOM...", state=tk.DISABLED)
                threading.Thread(target=engine.execute_manual_bottom_capture, args=(app.cap,app.txt_prod_name.get().strip(),"OP_LINE_01",wb,hb,gp,cx,cy,app.on_training_complete),daemon=True).start()

        cv2.rectangle(df,(x1,y1),(x2,y2),(255,255,255),1)
        cv2.line(df,(0,240),(640,240),(100,100,100),1,cv2.LINE_AA)
        cv2.line(df,(0,ly1),(640,ly1),(255,50,50),2,cv2.LINE_AA)
        cv2.putText(df,"LINE 1 TOP",(10,ly1-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,50,50),1)
        cv2.line(df,(0,ly2),(640,ly2),(50,50,255),2,cv2.LINE_AA)
        cv2.putText(df,"LINE 2 BOTTOM",(10,ly2+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(50,50,255),1)
        app.lbl_status.configure(text=ss, fg='#%02x%02x%02x' % (hc[2],hc[1],hc[0]))
        app.render_image_on_canvas(df)

# ══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    hw = HardwareController()
    engine = IndustrialSlateEngine(hw)
    app = BenchApp(engine)

    def start_bg_sequence():
        if not engine.is_holding_for_bg_sequence:
            engine.total_bg_slots_needed = int(app.txt_bg_target.get() if hasattr(app,'txt_bg_target') else 1)
            engine.background_stack = []; engine.current_bg_capture_index = 0
            engine.is_holding_for_bg_sequence = True; engine.trigger_retake_bg = False
            app.btn_init_bg.configure(text=f"CAPTURE STEP (1/{engine.total_bg_slots_needed})", bg=BTN_DANGER)
            app.lbl_bg_prompt.configure(text="→ Press again to capture", fg=FG_YELLOW)
        else:
            engine.trigger_retake_bg = True

    app.btn_init_bg.configure(command=start_bg_sequence)
    app.is_running = True
    t = threading.Thread(target=video_pipeline_worker, args=(app, engine), daemon=True)
    t.start()
    app.mainloop()
    app.is_running = False
    hw.close()
