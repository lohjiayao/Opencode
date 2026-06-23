from urllib import response

import cv2
import numpy as np
import time
import os
import torch
import serial
import serial.tools.list_ports
import torchvision.models as models
import torchvision.transforms as T
import threading

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import re
import json
import base64

# --- HARDWARE & AREA CONFIGURATIONS ---
BOX_X = 345
BOX_Y = 234
BOX_SCALE = 195
W_PCT = 0.94
H_PCT = 0.48
X_PCT = 0.5
Y_PCT = 0.48
ANALYSIS_THRESHOLD = 86

ORIENTATION_TOLERANCE = 2.0
PIXEL_SIZE_TOLERANCE = 47.0
DEEP_TEXTURE_THRESHOLD = 0.88
STABLE_FRAME_COUNT = 10

SMOOTHING_ALPHA = 0.15          
ANGLE_DEADBAND = 1.5           
STEPS_PER_REV = 1600.0         
DEGREES_PER_STEP = 360.0 / STEPS_PER_REV  
SERIAL_COOLDOWN_S = 1.5        

class IndustrialSlateEngine:
    def __init__(self):
        self.sm_angle = None
        self.last_transmission_time = 0.0
        self.target_angle_reference = None
        
        self.active_ui_tab = "INSPECTION" 
        self.is_inspection_active = False  
        self.stop_requested = False  
        
        # Multi-Stage Background Arrays
        self.background_stack = []
        self.current_bg_capture_index = 0
        self.total_bg_slots_needed = 1
        self.is_holding_for_bg_sequence = False
        
        # Multi-Product Signature Database (Max 5)
        self.product_database = {}
        self.active_product_key = None
        self.has_trained_product = False
        self.load_database_from_file()
        
        # General Last Cached Background
        self.background_memory = None  
        self.load_background_from_file()
        
        # Stability Verification Variables
        self.last_w, self.last_h, self.last_ang = 0, 0, 0
        self.stability_counter = 0
        self.is_profile_stable = False
        
        # Training Threading States
        self.is_currently_training = False
        self.training_progress_str = ""
        
        # Trigger Signals
        self.trigger_retake_bg = False
        self.trigger_save_settings = False
        self.trigger_record_product = False
        
        self.sharpen_kernel = np.array([[ 0, -0.5,  0],
                                        [-0.5,  3, -0.5],
                                        [ 0, -0.5,  0]])
        
        self.clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        
        self.ser = None
        self.auto_link_uart()

        print("[AI INIT] Loading ResNet-18 pipeline layers...")
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet.children())[:-1])
        self.feature_extractor.eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def auto_link_uart(self):
        # Force the connection directly to your known working port
        target_port = "COM8" 
        
        try:
            print(f"[SERIAL DIAGNOSTIC] Bypassing scan. Directly targeting: {target_port}...")
            
            # Close it first if it accidentally exists to prevent permission errors
            if hasattr(self, 'ser') and self.ser is not None and self.ser.is_open:
                self.ser.close()

            # Configure and open COM8 explicitly
            self.ser = serial.Serial(target_port, 9600, timeout=1.5)
            
            # Crucial: Give the Arduino Mega exactly 2 seconds to reboot its bootloader
            time.sleep(2) 
            
            # Flush the serial lines to clear bootup junk text
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            
            # Send the verification handshake parameter
            print(f"[SERIAL DIAGNOSTIC] Sending handshake: b'PING\\n'")
            self.ser.write(b"PING\n")
            
            # Read response line from Arduino
            raw_response = self.ser.readline()
            response = raw_response.decode('utf-8', errors='ignore').strip()
            
            print(f"[SERIAL DIAGNOSTIC] Raw response bytes received: {raw_response}")
            print(f"[SERIAL DIAGNOSTIC] Cleaned response text: '{response}'")
            
            if response == "PONG":
                print(f"[SERIAL SUCCESS] Handshake verified! Arduino Mega locked on {target_port}.")
                return True
            else:
                print(f"[SERIAL FAILURE] Port {target_port} opened, but responded with '{response}' instead of 'PONG'.")
                self.ser.close()
                self.ser = None
                return False

        except Exception as e:
            print(f"[SERIAL ERROR] Failed to bind to connection parameters on {target_port}: {e}")
            self.ser = None
            return False

        except Exception as e:
            print(f"[SERIAL ERROR] Failed to bind to connection parameters on {target_port}: {e}")
            self.ser = None
            return False

    def get_roi_coordinates(self, box_x, box_y, box_scale, w_pct, h_pct, x_pct, y_pct):
        half = box_scale // 2
        bx1 = max(0, min(box_x - half, 640 - box_scale))
        by1 = max(0, min(box_y - half, 480 - box_scale))
        
        rect_w, rect_h = int(640 * w_pct), int(480 * h_pct)
        start_x = int(x_pct * 640) - (rect_w // 2)
        start_y = int(y_pct * 480) - (rect_h // 2)
        
        x1, y1 = max(0, start_x), max(0, start_y)
        x2, y2 = min(640, start_x + rect_w), min(480, start_y + rect_h)
        return bx1, by1, box_scale, x1, y1, x2, y2

    def deskew_crop_roi(self, frame, cx, cy, w, h, angle):
        if frame is None or frame.size == 0:
            return np.zeros((48, 48, 3), dtype=np.uint8)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        crop = cv2.getRectSubPix(rotated, (max(1, int(w)), max(1, int(h))), (cx, cy))
        return crop

    def calculate_deep_vector(self, roi_bgr):
        with torch.no_grad():
            tensor = self.transform(roi_bgr).unsqueeze(0)
            vector = self.feature_extractor(tensor).flatten().numpy()
            return vector / np.linalg.norm(vector)

    def extract_micro_structure_signature(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return np.zeros(2)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
        enhanced = self.clahe.apply(gray)
        
        gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        mag, _ = cv2.cartToPolar(gx, gy)
        
        h, w = mag.shape
        mid = h // 2
        top_half_density = np.mean(mag[0:mid, :])
        bottom_half_density = np.mean(mag[mid:h, :])
        
        total = top_half_density + bottom_half_density + 1e-6
        return np.array([top_half_density / total, bottom_half_density / total])

    def load_database_from_file(self):
        if os.path.exists("product_db.json"):
            try:
                with open("product_db.json", "r") as f:
                    data = json.load(f)
                    for p_name, p_info in data.items():
                        p_info["fingerprints"] = [np.array(v) for v in p_info["fingerprints"]]
                        if "micro_signature" in p_info and p_info["micro_signature"]:
                            p_info["micro_signature"] = np.array(p_info["micro_signature"])
                        else:
                            p_info["micro_signature"] = None
                            
                        if "bg_bytes_base64" in p_info and p_info["bg_bytes_base64"]:
                            bg_bytes = base64.b64decode(p_info["bg_bytes_base64"])
                            np_arr = np.frombuffer(bg_bytes, dtype=np.uint8)
                            p_info["bg_image_matrix"] = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
                        else:
                            p_info["bg_image_matrix"] = None

                    self.product_database = data
                    if self.product_database:
                        self.active_product_key = list(self.product_database.keys())[0]
                        self.target_angle_reference = self.product_database[self.active_product_key]["angle_ref"]
                        self.has_trained_product = True
                        
                        if self.product_database[self.active_product_key]["bg_image_matrix"] is not None:
                            self.background_memory = self.product_database[self.active_product_key]["bg_image_matrix"]
            except Exception as e:
                print(f"[DB ERROR] Failed to parse local json profiles: {e}")

    def load_background_from_file(self):
        if self.background_memory is not None:
            return
        if os.path.exists("background_baseline.png"):
            try:
                self.background_memory = cv2.imread("background_baseline.png", cv2.IMREAD_GRAYSCALE)
            except Exception:
                pass

    def save_database_to_file(self):
        try:
            export_map = {}
            for p_name, p_info in self.product_database.items():
                bg_b64_str = ""
                if "bg_image_matrix" in p_info and p_info["bg_image_matrix"] is not None:
                    _, buffer = cv2.imencode('.png', p_info["bg_image_matrix"])
                    bg_b64_str = base64.b64encode(buffer).decode('utf-8')
                elif "bg_bytes_base64" in p_info:
                    bg_b64_str = p_info["bg_bytes_base64"]

                micro_sig_list = []
                if "micro_signature" in p_info and p_info["micro_signature"] is not None:
                    micro_sig_list = p_info["micro_signature"].tolist()

                export_map[p_name] = {
                    "operator_id": p_info["operator_id"],
                    "w_pixels": p_info["w_pixels"],
                    "h_pixels": p_info["h_pixels"],
                    "angle_ref": p_info["angle_ref"],
                    "bg_bytes_base64": bg_b64_str,
                    "micro_signature": micro_sig_list,
                    "fingerprints": [v.tolist() for v in p_info["fingerprints"]]
                }
            with open("product_db.json", "w") as f:
                json.dump(export_map, f, indent=4)
        except Exception as e:
            print(f"[DISK ERROR] Failed to parse json payload to file: {e}")

    def async_training_worker(self, cap, prod_name, op_id, current_w, current_h, box_x, box_y, box_scale, w_pct, h_pct, x_pct, y_pct, cx, cy, on_complete_callback):
        self.is_currently_training = True
        fingerprints = []
        micro_signatures = []
        
        captured = 0
        while captured < 50:
            ret, frame = cap.read()
            if not ret: continue
            
            bx1, by1, bs, x1, y1, x2, y2 = self.get_roi_coordinates(box_x, box_y, box_scale, w_pct, h_pct, x_pct, y_pct)
            zoom_crop = frame[by1:by1+bs, bx1:bx1+bs]
            processing_frame = cv2.resize(zoom_crop, (640, 480))
            
            roi_patch = self.deskew_crop_roi(processing_frame, cx, cy, current_w, current_h, self.sm_angle)
            
            if roi_patch.size > 0:
                vector = self.calculate_deep_vector(roi_patch)
                fingerprints.append(vector)
                
                micro_sig = self.extract_micro_structure_signature(roi_patch)
                micro_signatures.append(micro_sig)
                
                captured += 1
                self.training_progress_str = f"EXTRACTING FEATURES: {captured}/50 FRAMES..."
                time.sleep(0.01) 
                
        self.product_database[prod_name] = {
            "operator_id": op_id,
            "w_pixels": current_w,
            "h_pixels": current_h,
            "angle_ref": self.sm_angle,
            "bg_image_matrix": self.background_memory, 
            "micro_signature": np.mean(micro_signatures, axis=0),
            "fingerprints": fingerprints
        }
        self.active_product_key = prod_name
        self.target_angle_reference = self.sm_angle
        self.save_database_to_file()
        
        self.has_trained_product = True
        self.is_currently_training = False
        on_complete_callback()

    def identify_product_match(self, current_roi, current_w, current_h, size_tolerance, texture_threshold):
        if not self.active_product_key or self.active_product_key not in self.product_database: 
            return "MATCH", 1.0, ""
        
        active_model = self.product_database[self.active_product_key]
        current_vector = self.calculate_deep_vector(current_roi)
        
        active_scores = [np.dot(current_vector, target_vec) for target_vec in active_model["fingerprints"]]
        active_max_score = max(active_scores)
        
        w_drift = abs(current_w - active_model["w_pixels"])
        h_drift = abs(current_h - active_model["h_pixels"])
        
        if active_max_score > (texture_threshold - 0.12) and w_drift <= (size_tolerance + 20.0) and h_drift <= (size_tolerance + 20.0):
            if active_model.get("micro_signature") is not None:
                current_micro_sig = self.extract_micro_structure_signature(current_roi)
                
                trained_top_ratio = active_model["micro_signature"][0]
                live_top_ratio = current_micro_sig[0]
                
                if abs(live_top_ratio - trained_top_ratio) > 0.08:
                    return "REVERSE_SIDE", active_max_score, self.active_product_key
            
            return "MATCH", active_max_score, self.active_product_key
            
        for alternate_key, alternate_model in self.product_database.items():
            if alternate_key == self.active_product_key: 
                continue
            alt_scores = [np.dot(current_vector, target_vec) for target_vec in alternate_model["fingerprints"]]
            alt_max_score = max(alt_scores)
            
            alt_w_drift = abs(current_w - alternate_model["w_pixels"])
            alt_w_h_drift = abs(current_h - alternate_model["h_pixels"])
            
            if alt_max_score > texture_threshold and alt_w_drift <= size_tolerance and alt_w_h_drift <= size_tolerance:
                return "WRONG_SIDE", alt_max_score, alternate_key 
                
        return "WRONG_PRODUCT", 0.0, "" 


class BenchApp(tk.Tk):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.title("VISION CONTROL - HMI BENCH STATION")
        # FIXED: Expanded operational station dimensions to support expanded left panel footprint
        self.geometry("1250x700")
        self.configure(bg="#2b2b2b")
        
        self.cap = None
        self.parameter_values = {} 
        
        self.build_ui_layout()
        self.update_dropdown_lists()
        self.load_presets_index()
        
        self.notebook.select(self.tab_inspect)
        self.engine.active_ui_tab = "INSPECTION"
        
        self.is_running = True
        self.video_thread = threading.Thread(target=self.process_video_pipeline, daemon=True)
        self.video_thread.start()
        
    def build_ui_layout(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#2b2b2b', foreground='#ffffff')
        style.configure('TNotebook', background='#2b2b2b', borderwidth=0)
        style.configure('TNotebook.Tab', background='#3c3f41', foreground='#ffffff', padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', '#4b6eaf')])
        
        # FIXED: Extended the width parameter from 440 to 520 to completely clear clipped row items
        left_panel = tk.Frame(self, bg="#2b2b2b", width=520)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left_panel.pack_propagate(False)
        
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_train = tk.Frame(self.notebook, bg="#3c3f41")
        self.tab_inspect = tk.Frame(self.notebook, bg="#3c3f41")
        self.tab_calibrate = tk.Frame(self.notebook, bg="#3c3f41")
        
        self.notebook.add(self.tab_train, text="1. TRAINING")
        self.notebook.add(self.tab_inspect, text="2. INSPECTION")
        self.notebook.add(self.tab_calibrate, text="3. TUNING")
        
        self.setup_training_tab()
        self.setup_inspection_tab()
        self.setup_tuning_tab()
        
        right_panel = tk.Frame(self, bg="#212121")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.lbl_status = tk.Label(right_panel, text="CAMERA POWER OFF. PRESS START INSPECTION TO ATTACH STREAM.", bg="#212121", fg="#ffffff", font=("Helvetica", 11, "bold"))
        self.lbl_status.pack(anchor=tk.W, padx=10, pady=5)
        
        self.video_canvas = tk.Label(right_panel, bg="#000000")
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def setup_training_tab(self):
        tk.Label(self.tab_train, text="[STEP 1: TRACEABILITY ENTRY]", bg="#3c3f41", fg="#4b6eaf", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        
        tk.Label(self.tab_train, text="New Product Identity Name:", bg="#3c3f41", fg="#ffffff").pack(anchor=tk.W, padx=10, pady=2)
        self.txt_prod_name = tk.Entry(self.tab_train, bg="#2b2b2b", fg="#ffffff", insertbackground="white")
        self.txt_prod_name.pack(fill=tk.X, padx=10, pady=2)
        self.txt_prod_name.bind("<KeyRelease>", self.validate_training_workflow_state)
        
        tk.Label(self.tab_train, text="Operator Authorization ID:", bg="#3c3f41", fg="#ffffff").pack(anchor=tk.W, padx=10, pady=2)
        self.txt_op_id = tk.Entry(self.tab_train, bg="#2b2b2b", fg="#ffffff", insertbackground="white")
        self.txt_op_id.pack(fill=tk.X, padx=10, pady=2)
        self.txt_op_id.bind("<KeyRelease>", self.validate_training_workflow_state)
        
        ttk.Separator(self.tab_train, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
        tk.Label(self.tab_train, text="[STEP 2: BACKGROUND CALIBRATION]", bg="#3c3f41", fg="#4b6eaf", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        
        lbl_bg_count = tk.Label(self.tab_train, text="Background Variations Needed:", bg="#3c3f41", fg="#ffffff")
        lbl_bg_count.pack(anchor=tk.W, padx=10, pady=2)
        
        self.txt_bg_target = tk.Entry(self.tab_train, width=8, bg="#2b2b2b", fg="#ffffff", insertbackground="white")
        self.txt_bg_target.insert(0, "1") 
        self.txt_bg_target.pack(anchor=tk.W, padx=10, pady=2)
        
        self.btn_init_bg = tk.Button(self.tab_train, text="START BG CAPTURE STACK", bg="#424242", fg="#888888", state=tk.DISABLED, command=self.start_bg_stack_sequence)
        self.btn_init_bg.pack(fill=tk.X, padx=10, pady=5)
        
        self.lbl_bg_prompt = tk.Label(self.tab_train, text="Baseline Status: Locked. Complete Step 1.", bg="#3c3f41", fg="grey", font=("Helvetica", 9, "italic"))
        self.lbl_bg_prompt.pack(anchor=tk.W, padx=10, pady=2)
        
        ttk.Separator(self.tab_train, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
        tk.Label(self.tab_train, text="[STEP 3: SIGNATURE TRAINING]", bg="#3c3f41", fg="#4b6eaf", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=10, pady=5)
        
        self.btn_train_product = tk.Button(self.tab_train, text="TRAIN PRODUCT SIGNATURE", bg="#424242", fg="#888888", state=tk.DISABLED, command=self.trigger_product_training)
        self.btn_train_product.pack(fill=tk.X, padx=10, pady=12)
        
        ttk.Separator(self.tab_train, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
        self.btn_wipe_db = tk.Button(self.tab_train, text="⚠ CLEAR ENTIRE LOCAL DATABASE", bg="#b71c1c", fg="#ffffff", font=("Helvetica", 9, "bold"), command=self.wipe_product_database_handler)
        self.btn_wipe_db.pack(fill=tk.X, padx=10, pady=15)

    def wipe_product_database_handler(self):
        confirm = messagebox.askyesno("Database Master Wipe Confirmation", "Are you absolutely sure you want to delete all trained items from storage?\nThis cannot be undone.")
        if confirm:
            self.engine.product_database = {}
            self.engine.active_product_key = None
            self.engine.has_trained_product = False
            self.engine.target_angle_reference = None
            
            if os.path.exists("product_db.json"):
                try: os.remove("product_db.json")
                except Exception: pass
            
            self.update_dropdown_lists()
            self.validate_training_workflow_state()
            messagebox.showinfo("Database Reset", "Local signature memory cache completely wiped.")

    def validate_training_workflow_state(self, event=None):
        p_name = self.txt_prod_name.get().strip()
        o_id = self.txt_op_id.get().strip()
        
        if p_name and o_id and not self.engine.is_holding_for_bg_sequence:
            self.btn_init_bg.configure(state=tk.NORMAL, bg="#4b6eaf", fg="#ffffff")
            if self.engine.background_memory is None:
                self.lbl_bg_prompt.configure(text="Baseline Status: Ready for Step 2.", fg="yellow")
        else:
            self.btn_init_bg.configure(state=tk.DISABLED, bg="#424242", fg="#888888")
            self.btn_train_product.configure(state=tk.DISABLED, bg="#424242", fg="#888888")

    def setup_inspection_tab(self):
        lbl_info = tk.Label(self.tab_inspect, text="AUTOMATION INSPECTION ENGINE", bg="#3c3f41", fg="#ffffff", font=("Helvetica", 10, "bold"))
        lbl_info.pack(padx=10, pady=10)
        
        tk.Label(self.tab_inspect, text="Select Target Product Profile:", bg="#3c3f41", fg="#ffffff").pack(anchor=tk.W, padx=20, pady=2)
        self.cmb_inspect_products = ttk.Combobox(self.tab_inspect, state="readonly")
        self.cmb_inspect_products.pack(fill=tk.X, padx=20, pady=5)
        self.cmb_inspect_products.bind("<<ComboboxSelected>>", self.on_dropdown_model_changed)
        
        self.btn_start_inspect = tk.Button(self.tab_inspect, text="▶ START INSPECTION", bg="#2e7d32", fg="#ffffff", font=("Helvetica", 10, "bold"), command=self.start_inspection_routine)
        self.btn_start_inspect.pack(fill=tk.X, padx=20, pady=10)
        
        self.btn_stop_inspect = tk.Button(self.tab_inspect, text="■ STOP INSPECTION", bg="#b71c1c", fg="#ffffff", font=("Helvetica", 10, "bold"), command=self.stop_inspection_routine)
        self.btn_stop_inspect.pack(fill=tk.X, padx=20, pady=5)

        self.lbl_inspect_details = tk.Label(self.tab_inspect, text="Inspection Pipeline: CAMERA POWER OFF\nSelect profile and press Start.", bg="#3c3f41", fg="#b0bec5", justify=tk.LEFT)
        self.lbl_inspect_details.pack(anchor=tk.W, padx=20, pady=15)

    def setup_tuning_tab(self):
        scroll_canvas = tk.Canvas(self.tab_calibrate, bg="#3c3f41", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_calibrate, orient="vertical", command=scroll_canvas.yview)
        scroll_frame = tk.Frame(scroll_canvas, bg="#3c3f41")
        
        scroll_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        scroll_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")

        self.parameter_values = {
            cv2.CAP_PROP_EXPOSURE: -4 + 12,
            cv2.CAP_PROP_ZOOM: 100,
            cv2.CAP_PROP_FOCUS: 0,
            cv2.CAP_PROP_BRIGHTNESS: 109,
            cv2.CAP_PROP_CONTRAST: 97,
            cv2.CAP_PROP_SHARPNESS: 255,
            cv2.CAP_PROP_SATURATION: 255,
            "BOX_X": BOX_X, "BOX_Y": BOX_Y, "BOX_SCALE": BOX_SCALE,
            "W_PCT": int(W_PCT * 100), "H_PCT": int(H_PCT * 100),
            "X_PCT": int(X_PCT * 100), "Y_PCT": int(Y_PCT * 100),
            "THRESHOLD": ANALYSIS_THRESHOLD, "ORIENT": int(ORIENTATION_TOLERANCE),
            "SIZE": int(PIXEL_SIZE_TOLERANCE), "TEXTURE": int(DEEP_TEXTURE_THRESHOLD * 100),
            "STABLE_FRAMES": STABLE_FRAME_COUNT 
        }

        self.param_labels = {}

        def adjust_param(key, step, max_val, min_v=0):
            current_val = self.parameter_values[key]
            new_val = current_val + step
            if new_val > max_val: new_val = max_val
            if new_val < min_v: new_val = min_v
            self.parameter_values[key] = new_val
            self.param_labels[key].configure(text=str(new_val))

        def build_button_control_row(parent, label_text, key_id, max_v, min_v=0):
            row_frame = tk.Frame(parent, bg="#3c3f41")
            row_frame.pack(fill=tk.X, padx=15, pady=3)
            
            # FIXED: Expanded description text width boundary allocation from 22 to 25
            lbl_name = tk.Label(row_frame, text=label_text, bg="#3c3f41", fg="#ffffff", width=25, anchor=tk.W)
            lbl_name.pack(side=tk.LEFT)
            
            btn_m10 = tk.Button(row_frame, text="-10", width=4, bg="#424242", fg="#ffffff", command=lambda: adjust_param(key_id, -10, max_v, min_v))
            btn_m10.pack(side=tk.LEFT, padx=1)
            btn_m5 = tk.Button(row_frame, text="-5", width=4, bg="#424242", fg="#ffffff", command=lambda: adjust_param(key_id, -5, max_v, min_v))
            btn_m5.pack(side=tk.LEFT, padx=1)
            btn_m1 = tk.Button(row_frame, text="-1", width=3, bg="#424242", fg="#ffffff", command=lambda: adjust_param(key_id, -1, max_v, min_v))
            btn_m1.pack(side=tk.LEFT, padx=1)
            
            lbl_val = tk.Label(row_frame, text=str(self.parameter_values[key_id]), bg="#2b2b2b", fg="#00ff00", width=6, font=("Consolas", 10, "bold"))
            lbl_val.pack(side=tk.LEFT, padx=6)
            self.param_labels[key_id] = lbl_val
            
            btn_p1 = tk.Button(row_frame, text="+1", width=3, bg="#424242", fg="#ffffff", command=lambda: adjust_param(key_id, 1, max_v, min_v))
            btn_p1.pack(side=tk.LEFT, padx=1)
            btn_p5 = tk.Button(row_frame, text="+5", width=4, bg="#424242", fg="#ffffff", command=lambda: adjust_param(key_id, 5, max_v, min_v))
            btn_p5.pack(side=tk.LEFT, padx=1)
            btn_p10 = tk.Button(row_frame, text="+10", width=4, bg="#424242", fg="#ffffff", command=lambda: adjust_param(key_id, 10, max_v, min_v))
            btn_p10.pack(side=tk.LEFT, padx=1)

        tk.Label(scroll_frame, text="[CATEGORY 0: PRESET ENVIRONMENT MANAGEMENT]", bg="#3c3f41", fg="#4b6eaf", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=10, pady=6)
        preset_frame = tk.Frame(scroll_frame, bg="#3c3f41")
        preset_frame.pack(fill=tk.X, padx=15, pady=4)
        tk.Label(preset_frame, text="Active Environment Name:", bg="#3c3f41", fg="#ffffff").pack(side=tk.LEFT, padx=2)
        self.txt_preset_name = tk.Entry(preset_frame, width=12, bg="#2b2b2b", fg="#ffffff")
        self.txt_preset_name.insert(0, "Default_Line")
        self.txt_preset_name.pack(side=tk.LEFT, padx=4)
        
        self.cmb_presets = ttk.Combobox(preset_frame, width=14, state="readonly")
        self.cmb_presets.pack(side=tk.LEFT, padx=4)
        self.cmb_presets.bind("<<ComboboxSelected>>", self.on_preset_dropdown_changed)

        btn_load_p = tk.Button(preset_frame, text="LOAD", bg="#4b6eaf", fg="#ffffff", command=self.load_preset_to_ui)
        btn_load_p.pack(side=tk.LEFT, padx=2)
        btn_save_p = tk.Button(preset_frame, text="SAVE", bg="#2e7d32", fg="#ffffff", command=self.save_preset_to_json)
        btn_save_p.pack(side=tk.LEFT, padx=2)

        tk.Label(scroll_frame, text="[CATEGORY 1: CAMERA HARDWARE]", bg="#3c3f41", fg="#4b6eaf", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=10, pady=6)
        build_button_control_row(scroll_frame, "Exposure (Offset):", cv2.CAP_PROP_EXPOSURE, 12, 0)
        build_button_control_row(scroll_frame, "Zoom:", cv2.CAP_PROP_ZOOM, 500, 100)
        build_button_control_row(scroll_frame, "Focus Control (Manual):", cv2.CAP_PROP_FOCUS, 255, 0)
        build_button_control_row(scroll_frame, "Brightness:", cv2.CAP_PROP_BRIGHTNESS, 255, 0)
        build_button_control_row(scroll_frame, "Contrast:", cv2.CAP_PROP_CONTRAST, 255, 0)
        build_button_control_row(scroll_frame, "Sharpness:", cv2.CAP_PROP_SHARPNESS, 255, 0)
        build_button_control_row(scroll_frame, "Saturation:", cv2.CAP_PROP_SATURATION, 255, 0)

        tk.Label(scroll_frame, text="[CATEGORY 2: GEOMETRIC WINDOWS]", bg="#3c3f41", fg="#4b6eaf", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=10, pady=10)
        build_button_control_row(scroll_frame, "Box Center X (px):", "BOX_X", 640, 0)
        build_button_control_row(scroll_frame, "Box Center Y (px):", "BOX_Y", 480, 0)
        build_button_control_row(scroll_frame, "Box Scale (Zoom Size):", "BOX_SCALE", 640, 100)
        build_button_control_row(scroll_frame, "ROI Width %:", "W_PCT", 100, 0)
        build_button_control_row(scroll_frame, "ROI Height %:", "H_PCT", 100, 0)
        build_button_control_row(scroll_frame, "ROI Pos X %:", "X_PCT", 100, 0)
        build_button_control_row(scroll_frame, "ROI Pos Y %:", "Y_PCT", 100, 0)

        tk.Label(scroll_frame, text="[CATEGORY 3: THRESHOLD PROCESSING]", bg="#3c3f41", fg="#4b6eaf", font=("Helvetica", 9, "bold")).pack(anchor=tk.W, padx=10, pady=10)
        build_button_control_row(scroll_frame, "Binary Threshold Cutoff:", "THRESHOLD", 255, 0)
        build_button_control_row(scroll_frame, "Orientation Tolerance:", "ORIENT", 45, 0)
        build_button_control_row(scroll_frame, "Pixel Size Tolerance:", "SIZE", 200, 0)
        build_button_control_row(scroll_frame, "AI Deep Texture Match %:", "TEXTURE", 100, 50)
        build_button_control_row(scroll_frame, "Stability Window (Frames):", "STABLE_FRAMES", 100, 1)
            
        self.btn_save_all = tk.Button(scroll_frame, text="SAVE ALL PARAMETERS TO DISK", bg="#2e7d32", fg="#ffffff", font=("Helvetica", 10, "bold"), command=self.trigger_disk_save)
        self.btn_save_all.pack(fill=tk.X, padx=10, pady=25)

        # ======================================================================
        # DIRECT HARDWARE MANUAL OVERRIDE CONTROLS (GRID VIEW)
        # ======================================================================
        man_frame = ttk.LabelFrame(self.tab_train, text=" Manual Actuator Overrides ")
        man_frame.pack(fill="x", padx=10, pady=10)

        # Configure uniform grid columns
        man_frame.columnconfigure(0, weight=1)
        man_frame.columnconfigure(1, weight=1)

        def send_manual_cmd(command_string):
            if self.engine.ser and self.engine.ser.is_open:
                print(f"[MANUAL CONTROL] Dispatched command: {command_string}")
                self.engine.ser.reset_input_buffer()
                self.engine.ser.write(f"{command_string}\n".encode('utf-8'))
            else:
                messagebox.showwarning("Serial Connection Error", 
                                       "Arduino offline! Verify connection parameters under the Tuning tab.")

        # Row 0, Column 0: Clamp
        btn_clamp = ttk.Button(man_frame, text="Engage Clamp (Servo 0)", 
                               command=lambda: send_manual_cmd("MAN_CLAMP"))
        btn_clamp.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

        # Row 0, Column 1: Lift
        btn_lift = ttk.Button(man_frame, text="Elevate Hoist (Servos 1 & 2)", 
                              command=lambda: send_manual_cmd("MAN_LIFT"))
        btn_lift.grid(row=0, column=1, sticky="ew", padx=8, pady=6)

        # Row 1, Column 0: Flip
        btn_flip = ttk.Button(man_frame, text="Flip Profiles (Servos 3 & 4)", 
                             command=lambda: send_manual_cmd("MAN_FLIP"))
        btn_flip.grid(row=1, column=0, sticky="ew", padx=8, pady=6)

        # Row 1, Column 1: Reset All
        btn_release = ttk.Button(man_frame, text="Reset Deck (Retract & Open)", 
                                 command=lambda: send_manual_cmd("MAN_RELEASE"))
        btn_release.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        # ======================================================================

    def load_presets_index(self):
        if os.path.exists("environment_presets.json"):
            try:
                with open("environment_presets.json", "r") as f:
                    data = json.load(f)
                    self.cmb_presets.configure(values=list(data.keys()))
                    if data.keys(): self.cmb_presets.set(list(data.keys())[0])
            except Exception: pass

    def on_preset_dropdown_changed(self, event):
        self.txt_preset_name.delete(0, tk.END)
        self.txt_preset_name.insert(0, event.widget.get())

    def save_preset_to_json(self):
        name = self.txt_preset_name.get().strip()
        if not name: return
        presets = {}
        if os.path.exists("environment_presets.json"):
            try:
                with open("environment_presets.json", "r") as f: presets = json.load(f)
            except Exception: pass
        
        stringified_map = {str(k): v for k, v in self.parameter_values.items()}
        presets[name] = stringified_map
        try:
            with open("environment_presets.json", "w") as f: json.dump(presets, f, indent=4)
            self.load_presets_index()
            messagebox.showinfo("Preset Secure", f"Environment configuration '{name}' successfully compiled.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def load_preset_to_ui(self):
        name = self.cmb_presets.get()
        if not name or not os.path.exists("environment_presets.json"): return
        try:
            with open("environment_presets.json", "r") as f: presets = json.load(f)
            if name in presets:
                for k, v in presets[name].items():
                    try: target_key = int(k)
                    except ValueError: target_key = k
                    if target_key in self.parameter_values:
                        self.parameter_values[target_key] = v
                        if target_key in self.param_labels: self.param_labels[target_key].configure(text=str(v))
                messagebox.showinfo("Preset Active", f"Environment variables mapped into runtime scope.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def start_bg_stack_sequence(self):
        try:
            slots = int(self.txt_bg_target.get())
            if slots < 1: raise ValueError
            self.engine.total_bg_slots_needed = slots
            self.engine.background_stack = []
            self.engine.current_bg_capture_index = 0
            self.engine.is_holding_for_bg_sequence = True
            
            self.btn_init_bg.configure(text=f"CAPTURE STEP (1/{slots})", bg="#d32f2f")
            self.lbl_bg_prompt.configure(text=f"--> Action: Modify Tray for Position 1", fg="orange")
        except ValueError:
            messagebox.showerror("Configuration Error", "Please enter a valid positive integer for background variations.")

    def trigger_product_training(self):
        p_name = self.txt_prod_name.get().strip()
        o_id = self.txt_op_id.get().strip()
        
        if not p_name or not o_id:
            messagebox.showerror("Traceability Error", "Operator ID and Product Name fields cannot be left empty.")
            return
            
        if len(self.engine.product_database) >= 5 and p_name not in self.engine.product_database:
            overwrite_win = tk.Toplevel(self)
            overwrite_win.title("Database Maximum Capacity - Select Slot to Replace")
            overwrite_win.geometry("400x250")
            overwrite_win.configure(bg="#3c3f41")
            
            tk.Label(overwrite_win, text="Max 5 models reached.\nChoose a dataset entry profile to replace:", 
                     bg="#3c3f41", fg="#ffffff", font=("Helvetica", 10, "bold")).pack(pady=15)
            
            selected_slot = tk.StringVar()
            slot_dropdown = ttk.Combobox(overwrite_win, textvariable=selected_slot, values=list(self.engine.product_database.keys()), state="readonly")
            slot_dropdown.pack(pady=10)
            slot_dropdown.current(0)
            
            def execute_replacement():
                target_to_delete = selected_slot.get()
                if target_to_delete:
                    del self.engine.product_database[target_to_delete]
                    overwrite_win.destroy()
                    self.engine.trigger_record_product = True
                    
            tk.Button(overwrite_win, text="CONFIRM REPLACEMENT", bg="#b71c1c", fg="#ffffff", command=execute_replacement).pack(pady=15)
            return
            
        self.engine.trigger_record_product = True

    def on_training_complete(self):
        self.update_dropdown_lists()
        self.txt_prod_name.delete(0, tk.END)
        self.txt_op_id.delete(0, tk.END)
        self.validate_training_workflow_state()
        self.notebook.select(self.tab_inspect)
        self.btn_train_product.configure(text="TRAIN PRODUCT SIGNATURE", state=tk.DISABLED, bg="#424242", fg="#888888")
        messagebox.showinfo("AI Lock Secure", f"Product dataset compiled successfully into slot database profiles.")

    def start_inspection_routine(self):
        if not self.engine.active_product_key:
            messagebox.showerror("Execution Error", "No profile selected. Choose a trained dataset first.")
            return
            
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            SAVED_BRIGHTNESS = 110
            SAVED_CONTRAST = 100
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, SAVED_BRIGHTNESS)
            self.cap.set(cv2.CAP_PROP_CONTRAST, SAVED_CONTRAST)

        self.engine.stop_requested = False  
        self.engine.is_inspection_active = True
        self.lbl_inspect_details.configure(text=f"Inspection Pipeline: RUNNING\nActive Profile: {self.engine.active_product_key}\nHardware Steppers Connected.", fg="#00ff00")

    def stop_inspection_routine(self):
        self.engine.is_inspection_active = False
        self.engine.stop_requested = True  
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        black_block = np.zeros((480, 640, 3), dtype=np.uint8)
        self.render_image_on_canvas(black_block)
        
        self.lbl_inspect_details.configure(text="Inspection Pipeline: CAMERA POWER OFF\nConveyor logic safe and inert.", fg="#b0bec5")
        self.lbl_status.configure(text="CAMERA INERT. PRESS START INSPECTION TO ACTIVE DATA LINKS.", fg="#ffffff")

    def on_dropdown_model_changed(self, event):
        selected_model = event.widget.get()
        self.engine.active_product_key = selected_model
        if selected_model in self.engine.product_database:
            self.engine.target_angle_reference = self.engine.product_database[selected_model]["angle_ref"]
            if self.engine.product_database[selected_model]["bg_image_matrix"] is not None:
                self.engine.background_memory = self.engine.product_database[selected_model]["bg_image_matrix"]
        self.update_dropdown_lists()

    def update_dropdown_lists(self):
        models_available = list(self.engine.product_database.keys())
        self.cmb_inspect_products.configure(values=models_available)
        if self.engine.active_product_key:
            self.cmb_inspect_products.set(self.engine.active_product_key)

    def trigger_disk_save(self):
        cur_exp = self.parameter_values[cv2.CAP_PROP_EXPOSURE] - 12
        cur_zoom = self.parameter_values[cv2.CAP_PROP_ZOOM]
        cur_focus = self.parameter_values[cv2.CAP_PROP_FOCUS]
        cur_bright = self.parameter_values[cv2.CAP_PROP_BRIGHTNESS]
        cur_contrast = self.parameter_values[cv2.CAP_PROP_CONTRAST]
        cur_sharp = self.parameter_values[cv2.CAP_PROP_SHARPNESS]
        cur_sat = self.parameter_values[cv2.CAP_PROP_SATURATION]
        
        c_bx = self.parameter_values["BOX_X"]
        c_by = self.parameter_values["BOX_Y"]
        c_bs = self.parameter_values["BOX_SCALE"]
        c_wp = self.parameter_values["W_PCT"] / 100.0
        c_hp = self.parameter_values["H_PCT"] / 100.0
        c_xp = self.parameter_values["X_PCT"] / 100.0
        c_yp = self.parameter_values["Y_PCT"] / 100.0
        
        c_th = self.parameter_values["THRESHOLD"]
        c_or = float(self.parameter_values["ORIENT"])
        c_sz = float(self.parameter_values["SIZE"])
        c_tx = self.parameter_values["TEXTURE"] / 100.0
        c_sf = int(self.parameter_values["STABLE_FRAMES"])

        try:
            script_path = __file__
            with open(script_path, "r", encoding="utf-8") as file:
                code = file.read()
                
            code = re.sub(r"BOX_X = \d+", f"BOX_X = {c_bx}", code)
            code = re.sub(r"BOX_Y = \d+", f"BOX_Y = {c_by}", code)
            code = re.sub(r"BOX_SCALE = \d+", f"BOX_SCALE = {c_bs}", code)
            code = re.sub(r"W_PCT = \d+\.\d+", f"W_PCT = {round(c_wp, 2)}", code)
            code = re.sub(r"H_PCT = \d+\.\d+", f"H_PCT = {round(c_hp, 2)}", code)
            code = re.sub(r"X_PCT = \d+\.\d+", f"X_PCT = {round(c_xp, 2)}", code)
            code = re.sub(r"Y_PCT = \d+\.\d+", f"Y_PCT = {round(c_yp, 2)}", code)
            code = re.sub(r"ANALYSIS_THRESHOLD = \d+", f"ANALYSIS_THRESHOLD = {c_th}", code)
            code = re.sub(r"STABLE_FRAME_COUNT = \d+", f"STABLE_FRAME_COUNT = {c_sf}", code)
            
            code = re.sub(r"ORIENTATION_TOLERANCE = \d+\.\d+", f"ORIENTATION_TOLERANCE = {c_or}", code)
            code = re.sub(r"PIXEL_SIZE_TOLERANCE = \d+\.\d+", f"PIXEL_SIZE_TOLERANCE = {c_sz}", code)
            code = re.sub(r"DEEP_TEXTURE_THRESHOLD = \d+\.\d+", f"DEEP_TEXTURE_THRESHOLD = {c_tx}", code)
            
            code = re.sub(r"SAVED_EXPOSURE = -\d+|SAVED_EXPOSURE = \d+", f"SAVED_EXPOSURE = {cur_exp}", code)
            code = re.sub(r"SAVED_ZOOM = \d+", f"SAVED_ZOOM = {cur_zoom}", code)
            code = re.sub(r"SAVED_FOCUS = \d+", f"SAVED_FOCUS = {cur_focus}", code)
            code = re.sub(r"SAVED_BRIGHTNESS = \d+", f"SAVED_BRIGHTNESS = {cur_bright}", code)
            code = re.sub(r"SAVED_CONTRAST = \d+", f"SAVED_CONTRAST = {cur_contrast}", code)
            code = re.sub(r"SAVED_SHARPNESS = \d+", f"SAVED_SHARPNESS = {cur_sharp}", code)
            code = re.sub(r"SAVED_SATURATION = \d+", f"SAVED_SATURATION = {cur_sat}", code)

            with open(script_path, "w", encoding="utf-8") as file:
                file.write(code)
            messagebox.showinfo("Disk Burn Success", "All camera parameters and configuration profiles updated on disk.")
        except Exception as e:
            messagebox.showerror("Disk Error", f"Could not sync variables: {e}")

    def process_video_pipeline(self):
        time.sleep(0.5)
        
        while self.is_running:
            if self.engine.stop_requested:
                time.sleep(0.05)
                continue

            try:
                raw_tab_text = self.notebook.tab(self.notebook.select(), "text")
                tab_name = raw_tab_text.split(".")[-1].strip().upper()
                self.engine.active_ui_tab = tab_name
            except Exception:
                tab_name = "TRAINING"
                self.engine.active_ui_tab = tab_name

            needs_camera = (tab_name in ["TRAINING", "TUNING"]) or (tab_name == "INSPECTION" and self.engine.is_inspection_active)
            
            if not needs_camera:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                if tab_name == "INSPECTION":
                    black_block = np.zeros((480, 640, 3), dtype=np.uint8)
                    self.render_image_on_canvas(black_block)
                time.sleep(0.05)
                continue
                
            if self.cap is None or not self.cap.isOpened():
                if self.engine.active_ui_tab in ["TRAINING", "TUNING"] or self.engine.is_inspection_active:
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                else:
                    time.sleep(0.05)
                    continue

            l_bx = self.parameter_values["BOX_X"]
            l_by = self.parameter_values["BOX_Y"]
            l_bs = self.parameter_values["BOX_SCALE"]
            l_wp = self.parameter_values["W_PCT"] / 100.0
            l_hp = self.parameter_values["H_PCT"] / 100.0
            l_xp = self.parameter_values["X_PCT"] / 100.0
            l_yp = self.parameter_values["Y_PCT"] / 100.0
            
            live_threshold_limit = self.parameter_values["THRESHOLD"]
            live_orient_tol = float(self.parameter_values["ORIENT"])
            live_size_tol = float(self.parameter_values["SIZE"])
            live_texture_limit = self.parameter_values["TEXTURE"] / 100.0
            target_limit_window = self.parameter_values["STABLE_FRAMES"]

            if self.engine.active_ui_tab == "TUNING" and self.cap is not None:
                for prop in [cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_ZOOM, cv2.CAP_PROP_FOCUS, cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_CONTRAST, cv2.CAP_PROP_SHARPNESS, cv2.CAP_PROP_SATURATION]:
                    val = self.parameter_values[prop]
                    if prop == cv2.CAP_PROP_EXPOSURE:
                        self.cap.set(prop, val - 12)
                    else:
                        self.cap.set(prop, val)

            ret, frame = self.cap.read()
            if not ret: continue

            bx1, by1, bs, x1, y1, x2, y2 = self.engine.get_roi_coordinates(l_bx, l_by, l_bs, l_wp, l_hp, l_xp, l_yp)
            zoom_crop = frame[by1:by1+bs, bx1:bx1+bs]
            display_frame = cv2.resize(zoom_crop, (640, 480))
            
            masked_processing_frame = np.zeros_like(display_frame)
            masked_processing_frame[y1:y2, x1:x2] = display_frame[y1:y2, x1:x2]
            masked_processing_frame = cv2.filter2D(masked_processing_frame, -1, self.engine.sharpen_kernel)

            if tab_name == "TRAINING":
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
                
                if self.engine.is_holding_for_bg_sequence:
                    gray_frame = cv2.cvtColor(masked_processing_frame, cv2.COLOR_BGR2GRAY)
                    blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
                    
                    if self.engine.trigger_retake_bg:
                        self.engine.trigger_retake_bg = False
                        self.engine.background_stack.append(blur_frame)
                        self.engine.current_bg_capture_index += 1
                        
                        if self.engine.current_bg_capture_index >= self.engine.total_bg_slots_needed:
                            self.engine.background_memory = np.mean(self.engine.background_stack, axis=0).astype(np.uint8)
                            cv2.imwrite("background_baseline.png", self.engine.background_memory)
                            self.engine.is_holding_for_bg_sequence = False
                            self.btn_init_bg.configure(text="START BG CAPTURE STACK", bg="#4b6eaf")
                            self.lbl_bg_prompt.configure(text=f"✔ Baselines Saved ({len(self.engine.background_stack)} Slots)", fg="green")
                            self.btn_train_product.configure(state=tk.NORMAL, bg="#2e7d32", fg="#ffffff")
                        else:
                            idx = self.engine.current_bg_capture_index
                            tot = self.engine.total_bg_slots_needed
                            self.btn_init_bg.configure(text=f"CAPTURE STEP ({idx+1}/{tot})")
                            self.lbl_bg_prompt.configure(text=f"--> Action: Modify Tray for Position 1", fg="orange")
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    self.lbl_status.configure(text="HOLD: STEPPERS INERT | CAPTURING STORAGE LAYERS...", fg="orange")
                    self.render_image_on_canvas(display_frame)
                    continue

                if self.engine.is_currently_training:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    self.lbl_status.configure(text=self.engine.training_progress_str, fg="orange")
                    self.render_image_on_canvas(display_frame)
                    continue

                if self.engine.trigger_retake_bg:
                    self.engine.trigger_retake_bg = False
                    self.start_bg_stack_sequence()
                    continue

                gray_frame = cv2.cvtColor(masked_processing_frame, cv2.COLOR_BGR2GRAY)
                blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
                if self.engine.background_memory is not None:
                    delta_map = cv2.absdiff(self.engine.background_memory, blur_frame)
                else:
                    delta_map = blur_frame
                _, thresh_mask = cv2.threshold(delta_map, live_threshold_limit, 255, cv2.THRESH_BINARY)
                analysis_mask = np.zeros_like(thresh_mask)
                analysis_mask[y1:y2, x1:x2] = thresh_mask[y1:y2, x1:x2]
                cnts, _ = cv2.findContours(analysis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_contour = None
                max_area = 0
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area > max_area and area > 100:
                        max_area = area
                        best_contour = c

                if best_contour is not None:
                    rect = cv2.minAreaRect(best_contour)
                    (cx, cy), (w_box, h_box), raw_angle = rect
                    if w_box < h_box:
                        w_box, h_box = h_box, w_box
                        raw_angle += 90.0

                    w_delta = abs(w_box - self.engine.last_w)
                    h_delta = abs(h_box - self.engine.last_h)
                    a_delta = abs(raw_angle - self.engine.last_ang)
                    if w_delta < 1.5 and h_delta < 1.5 and a_delta < 1.5:
                        self.engine.stability_counter += 1
                    else:
                        self.engine.stability_counter = 0
                        self.engine.is_profile_stable = False
                    self.engine.last_w, self.engine.last_h, self.engine.last_ang = w_box, h_box, raw_angle
                    
                    if self.engine.stability_counter >= target_limit_window:
                        self.engine.is_profile_stable = True
                        if self.engine.sm_angle is None: self.engine.sm_angle = raw_angle
                    
                    if self.engine.is_profile_stable:
                        status_str = f"TRAINING PREVIEW | PART STATIC AND LOCKED | READY TO TRAIN"
                        hud_color = (0, 255, 0)
                    else:
                        status_str = f"TRAINING PREVIEW | STABILIZING PART ({self.engine.stability_counter}/{target_limit_window})"
                        hud_color = (0, 165, 255)
                    
                    local_box_pts = cv2.boxPoints(((cx, cy), (w_box, h_box), raw_angle)).astype(np.int32)
                    cv2.drawContours(display_frame, [local_box_pts], 0, hud_color, 1)
                else:
                    self.engine.stability_counter = 0
                    self.engine.is_profile_stable = False
                    status_str = "TRAINING PREVIEW | PLACE PRODUCT INSIDE THE WINDOW"
                
                if self.engine.trigger_record_product:
                    self.engine.trigger_record_product = False
                    if best_contour is not None and self.engine.is_profile_stable:
                        self.btn_train_product.configure(text="PROCESSING...", state=tk.DISABLED)
                        self.engine.target_angle_reference = self.engine.sm_angle
                        p_name = self.txt_prod_name.get().strip()
                        o_id = self.txt_op_id.get().strip()
                        t = threading.Thread(
                            target=self.engine.async_training_worker, 
                            args=(self.cap, p_name, o_id, w_box, h_box, l_bx, l_by, l_bs, l_wp, l_hp, l_xp, l_yp, cx, cy, self.on_training_complete),
                            daemon=True
                        )
                        t.start()
                    else:
                        messagebox.showwarning("Stability Warning", f"Component must be static for {target_limit_window} frames before training loop dispatches.")

                self.lbl_status.configure(text=status_str, fg="#ffffff")
                self.render_image_on_canvas(display_frame)
                continue

            if self.engine.background_memory is not None or self.engine.has_trained_product:
                gray_frame = cv2.cvtColor(masked_processing_frame, cv2.COLOR_BGR2GRAY)
                blur_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
                
                if self.engine.background_memory is not None:
                    delta_map = cv2.absdiff(self.engine.background_memory, blur_frame)
                else:
                    delta_map = blur_frame
                    
                _, thresh_mask = cv2.threshold(delta_map, live_threshold_limit, 255, cv2.THRESH_BINARY)
                analysis_mask = np.zeros_like(thresh_mask)
                analysis_mask[y1:y2, x1:x2] = thresh_mask[y1:y2, x1:x2]
                cnts, _ = cv2.findContours(analysis_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                best_contour = None
                max_area = 0
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area > max_area and area > 100:
                        max_area = area
                        best_contour = c
            else:
                best_contour = None

            if best_contour is not None:
                rect = cv2.minAreaRect(best_contour)
                (cx, cy), (w_box, h_box), raw_angle = rect
                if w_box < h_box:
                    w_box, h_box = h_box, w_box
                    raw_angle += 90.0

                w_delta = abs(w_box - self.engine.last_w)
                h_delta = abs(h_box - self.engine.last_h)
                a_delta = abs(raw_angle - self.engine.last_ang)
                
                if w_delta < 1.5 and h_delta < 1.5 and a_delta < 1.5:
                    self.engine.stability_counter += 1
                else:
                    self.engine.stability_counter = 0
                    self.engine.is_profile_stable = False
                    
                self.engine.last_w, self.engine.last_h, self.engine.last_ang = w_box, h_box, raw_angle
                
                if self.engine.stability_counter >= target_limit_window:
                    self.engine.is_profile_stable = True

                if self.engine.is_profile_stable:
                    if self.engine.sm_angle is None: self.engine.sm_angle = raw_angle
                    else:
                        angle_delta = raw_angle - self.engine.sm_angle
                        if angle_delta > 180: angle_delta -= 360
                        elif angle_delta < -180: angle_delta += 360
                        if abs(angle_delta) > ANGLE_DEADBAND:
                            self.engine.sm_angle += (SMOOTHING_ALPHA * angle_delta)

                    roi_patch = self.engine.deskew_crop_roi(masked_processing_frame, cx, cy, w_box, h_box, self.engine.sm_angle)
                    match_status, text_score, matched_series_name = self.engine.identify_product_match(roi_patch, w_box, h_box, live_size_tol, live_texture_limit)

                    # # ======================================================================
                    # # ARDUINO STEPPER & SERVO SERIAL HANDSHAKE
                    # # ======================================================================
                    # if match_status == "MATCH" and self.engine.is_inspection_active:
                    #     if self.engine.ser and self.engine.ser.is_open:
                    #         print("[VISION MASTER] Target confirmed! Dispatching HALT and TRIGGER payloads...")
                            
                    #         # Wipes any leftover serial data to guarantee immediate delivery
                    #         self.engine.ser.reset_input_buffer()
                            
                    #         # Send 'START\n' to freeze the stepper and trigger the servos
                    #         self.engine.ser.write(b"START\n")
                            
                    #         print("[VISION MASTER] Communications locked. Waiting for hardware 'DONE' flag...")
                            
                    #         # Keep Python loop frozen here while physical servos work
                    #         while self.engine.is_inspection_active:
                    #             if self.engine.ser.in_waiting > 0:
                    #                 response = self.engine.ser.readline().decode('utf-8').strip()
                    #                 if response == "DONE":
                    #                     print("[VISION MASTER] Handshake confirmed. Resuming conveyor flow...")
                    #                     break
                    #             time.sleep(0.01)
                    # # ======================================================================

                    # FIXED: Calculate angle drift parameters even if match state is REVERSE_SIDE
                    spin_needed = 0.0
                    if self.engine.target_angle_reference is not None:
                        rot_delta = self.engine.sm_angle - self.engine.target_angle_reference
                        if rot_delta > 180: rot_delta -= 360
                        elif rot_delta < -180: rot_delta += 360
                        spin_needed = round(rot_delta, 1)

                    if self.engine.is_inspection_active and self.engine.active_product_key and match_status == "MATCH" and abs(spin_needed) > live_orient_tol:
                        self.engine.send_step_command(spin_needed)

                    if not self.engine.active_product_key:
                        status_str = f"PRODUCT SPOTTED ({int(w_box)}x{int(h_box)}px) - SELECT ACTIVE DATASET PROFILE SLOT"
                        hud_color = (0, 255, 255) 
                    elif match_status == "REVERSE_SIDE":
                        # FIXED: Merged structural alignment status reports inside inversion tracking blocks
                        status_str = f"REJECT ANOMALY: REVERSE SIDE DETECTED | Delta: {spin_needed} deg"
                        hud_color = (255, 191, 0)  # Light Blue
                    elif match_status == "WRONG_SIDE":
                        status_str = f"REJECT ANOMALY: DIFFERENT SERIES DETECTED (Detected Series: '{matched_series_name}')"
                        hud_color = (30, 75, 120)  # Industrial Brown
                    elif match_status == "WRONG_PRODUCT":
                        status_str = f"CRITICAL REJECTION: HARD SIZE OR PATTERN MISMATCH ({int(w_box)}x{int(h_box)}px)"
                        hud_color = (0, 0, 255)  
                    else:
                        if self.engine.target_angle_reference is not None:
                            if abs(spin_needed) <= live_orient_tol:
                                status_str = f"ACTIVE MATCH: '{self.engine.active_product_key}' | PERFECTLY ALIGNED | Delta: {spin_needed} deg"
                                hud_color = (0, 255, 0)
                            else:
                                status_str = f"ACTIVE MATCH: '{self.engine.active_product_key}' | MISALIGNED DEVIATION | Delta: {spin_needed} deg"
                                hud_color = (0, 255, 255) 
                        else:
                            status_str = f"PRODUCT MATCHED | Mapping Initial Orientation Baseline..."
                            hud_color = (0, 255, 0)

                    local_box_pts = cv2.boxPoints(((cx, cy), (w_box, h_box), self.engine.sm_angle)).astype(np.int32)
                    cv2.drawContours(display_frame, [local_box_pts], 0, hud_color, 2)
                else:
                    status_str = f"PRODUCT DETECTED... VERIFYING COGNITIVE STABILITY ({self.engine.stability_counter}/{target_limit_window})"
                    hud_color = (0, 165, 255)
            else:
                self.engine.sm_angle = None
                self.engine.stability_counter = 0
                self.engine.is_profile_stable = False
                if self.engine.background_memory is None:
                    status_str = f"SYSTEM INITIALIZED | ACTIVE PROFILE: '{self.engine.active_product_key}' | READY TO DEPLOY"
                    hud_color = (0, 255, 0)
                else:
                    status_str = f"PIPELINE ACTIVE Mode: {self.engine.active_ui_tab} | SYSTEM IDLE"
                    hud_color = (0, 255, 0)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            self.lbl_status.configure(text=status_str, fg='#%02x%02x%02x' % (hud_color[2], hud_color[1], hud_color[0]))
            self.render_image_on_canvas(display_frame)

    def render_image_on_canvas(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_canvas.configure(image=img_tk)
        self.video_canvas.image = img_tk

    def destroy(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        super().destroy()


if __name__ == "__main__":
    core_engine = IndustrialSlateEngine()
    app = BenchApp(core_engine)
    
    def tk_click_interceptor():
        core_engine.trigger_retake_bg = True
        
    app.btn_init_bg.configure(command=tk_click_interceptor)
    
    # Apply initial hardware defaults
    SAVED_EXPOSURE = -4
    SAVED_ZOOM = 100
    SAVED_FOCUS = 13
    SAVED_BRIGHTNESS = 110
    SAVED_CONTRAST = 100
    SAVED_SHARPNESS = 255
    SAVED_SATURATION = 255
    
    app.mainloop()