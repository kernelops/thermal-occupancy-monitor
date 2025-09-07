########################################
# Authors
# 1. Nishant V H
# 2. Suhas Papanashi
# 3. Manojith Bhat 
# RV College of Engineering, Bengaluru, India
########################################
import cv2
import numpy as np
import time
import math
import threading
import csv
import datetime
import os
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer

########################################
# Custom Colormap Function
########################################

def create_custom_thermal_colormap_custom():
    """
    Creates a custom colormap with the following progression:
      - 0 to 50: Solid white (255,255,255).
      - 51 to 100: Transition from white (255,255,255) to pure cyan (255,255,0).
      - 101 to 150: Transition from pure cyan (255,255,0) to darker cyan (150,150,0).
      - 151 to 200: Transition from darker cyan (150,150,0) to pink (230,130,200).
      - 201 to 255: Transition from pink (230,130,200) to darkest deep pink (147,20,255).
    """
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    
    for i in range(0, 51):
        colormap[i] = [255, 255, 255]
    
    for i in range(51, 101):
        t = (i - 51) / (100 - 51)
        r = int(255 * (1 - t) + 0 * t)
        colormap[i] = [255, 255, r]
    
    for i in range(101, 151):
        t = (i - 101) / (150 - 101)
        b = int(255 * (1 - t) + 150 * t)
        g = int(255 * (1 - t) + 150 * t)
        r = 0
        colormap[i] = [b, g, r]
    
    for i in range(151, 201):
        t = (i - 151) / (200 - 151)
        b = int(150 * (1 - t) + 230 * t)
        g = int(150 * (1 - t) + 130 * t)
        r = int(0 * (1 - t) + 200 * t)
        colormap[i] = [b, g, r]

    for i in range(201, 256):
        t = (i - 201) / (255 - 201)
        b = int(230 * (1 - t) + 147 * t)
        g = int(130 * (1 - t) + 20 * t)
        r = int(200 * (1 - t) + 255 * t)
        colormap[i] = [b, g, r]
    
    return colormap

custom_cmap1 = create_custom_thermal_colormap_custom()

########################################
# Global variables for sensor data
########################################

latest_thermal_data = np.zeros((8, 8), dtype=np.float32)
latest_dht_temp = 0.0

########################################
# CSV Data Export Setup Functions
########################################

def get_next_run_number(base_filename="data"):
    """
    Scans the current directory for files named 'data_*.csv',
    finds the highest existing number, and returns the next number.
    For example, if 'data_1.csv' and 'data_2.csv' exist, this returns 3.
    If no such files exist, it returns 1.
    """
    existing_nums = []
    for f in os.listdir('.'):
        if f.startswith(f"{base_filename}_") and f.endswith(".csv"):
            try:
                # Extract the number part of the filename
                num_str = f[len(base_filename)+1 : -4]
                existing_nums.append(int(num_str))
            except ValueError:
                # Ignore files that don't have a valid number, e.g., 'data_test.csv'
                continue
    
    if not existing_nums:
        return 1
    else:
        return max(existing_nums) + 1

# --- Initialization Code ---
# 1. Determine the filename for this specific run
run_number = get_next_run_number()
log_filename = f"data_{run_number}.csv"
print(f"Logging data for this run to: {log_filename}")

# 2. Set up the global variables for the CSV writer
record_id = 0
csv_file = open(log_filename, "a", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["record_id", "timestamp", "dht_temperature", "gridEye_array", "current_people_count", "current_objects_count"])


def export_data(sensor_frame, dht_temp, current_people, current_objects):
    global record_id
    record_id += 1
    timestamp = datetime.datetime.now().isoformat()
    grid_values = sensor_frame.flatten().tolist()
    row = [record_id, timestamp, dht_temp, grid_values, current_people, current_objects]
    csv_writer.writerow(row)
    csv_file.flush()

########################################
# Flask Server to Receive Sensor Data
########################################

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def handle_data():
    data = request.get_json()
    if data is None: return jsonify({"error": "Invalid JSON data"}), 400
    grid_data = data.get("gridEye")
    if grid_data is None or len(grid_data) != 64: return jsonify({"error": "Missing or invalid gridEye data"}), 400
    global latest_thermal_data
    latest_thermal_data = np.array(grid_data, dtype=np.float32).reshape((8,8))
    dht_data = data.get("dht")
    if dht_data is None or "temperature" not in dht_data: return jsonify({"error": "Missing DHT temperature data"}), 400
    try:
        global latest_dht_temp
        latest_dht_temp = float(dht_data["temperature"]) + 0.1
    except ValueError: return jsonify({"error": "Invalid DHT temperature value"}), 400
    return jsonify({"status": "success"}), 200

@app.route('/')
def index(): return "Flask server with Gevent is running!"

def run_flask_server():
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

########################################
# 1. Data Upscaling
########################################

def interpolate8to71(input_array):
    h, w = input_array.shape
    out = np.zeros((71, 71), dtype=np.float32)
    for r in range(h):
        for c in range(w): out[10 * r, 10 * c] = input_array[r, c]
        for c in range(w - 1):
            left, right = input_array[r, c], input_array[r, c + 1]
            diff = right - left
            for newcol in range(1, 10): out[10 * r, 10 * c + newcol] = left + (diff * newcol / 10)
    for r in range(h - 1):
        for c in range(71):
            up, down = out[10 * r, c], out[10 * (r + 1), c]
            diff = down - up
            for newrow in range(1, 10): out[10 * r + newrow, c] = up + (diff * newrow / 10)
    return out

########################################
# 2. Preprocessing and Blob Detection
########################################

def threshold_image(img, dht_temp_thresh): return np.uint8(img >= dht_temp_thresh)
def connected_components(mask): return cv2.connectedComponents(mask)

########################################
# 3. Feature Extraction
########################################

def compute_blob_features(labels, label, img_shape):
    blob_mask = (labels == label).astype(np.uint8)
    coords = np.column_stack(np.where(blob_mask == 1))
    if coords.size == 0: return None
    size = coords.shape[0]
    centroid = np.mean(coords, axis=0)
    dt_blob = compute_distance_transform(blob_mask)
    central_coords_blob, _, _ = detect_central_points(dt_blob)
    return blob_mask, size, centroid, central_coords_blob

def compute_distance_transform(blob_mask): return cv2.distanceTransform(blob_mask * 255, distanceType=cv2.DIST_L1, maskSize=3).astype(np.float32)

def detect_central_points(dt, value_threshold=3, count_threshold=6):
    kernel = np.array([[-1, -2, -1], [-2, 12, -2], [-1, -2, -1]], dtype=np.int32)
    conv = cv2.filter2D(dt, -1, kernel)
    central_mask = ((dt > value_threshold) & (conv >= count_threshold)).astype(np.uint8)
    coords = np.column_stack(np.where(central_mask == 1))
    values = dt[central_mask == 1]
    return coords, values, central_mask

########################################
# 4. Multi-Object Tracking
########################################

tracked_objects = []
MATCH_THRESHOLD_CENTROID = 25
CENTRAL_POINT_DIST_THRESHOLD = 5
JACCARD_THRESHOLD_CP = 0.3

class HumanObject:
    def __init__(self, label, pos, size, central_points):
        self.label = label
        self.pos = pos
        self.size = size
        self.central_points = central_points
        self.path = [pos]
        self.virtual_age = 0
        self.updated = True
    
    def update(self, new_pos, new_size, new_central_points):
        self.pos = new_pos
        self.size = new_size
        self.central_points = new_central_points
        self.path.append(new_pos)
        self.virtual_age = 0
        self.updated = True
    
    def virtual_propagate(self):
        self.virtual_age += 1
        self.updated = False

def calculate_jaccard_index_cp(set1_cp, set2_cp, dist_thresh):
    if not set1_cp.size or not set2_cp.size: return 0.0
    matches = 0
    list1_cp = [tuple(p) for p in set1_cp]
    list2_cp = [tuple(p) for p in set2_cp]
    used_set2_indices = set()
    for p1 in list1_cp:
        for idx2, p2 in enumerate(list2_cp):
            if idx2 not in used_set2_indices:
                distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if distance < dist_thresh:
                    matches += 1
                    used_set2_indices.add(idx2)
                    break
    union_size = len(list1_cp) + len(list2_cp) - matches
    if union_size == 0: return 1.0 if matches > 0 else 0.0
    return matches / union_size

########################################
# Normalization Function using Baseline
########################################

def normalize_with_baseline(img, baseline_temp):
    min_val, max_val = baseline_temp, img.max()
    if max_val <= min_val: return np.zeros_like(img, dtype=np.uint8)
    return ((img - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)

########################################
# 7. Integrated Visualization Dashboard
########################################

def create_combined_visualization(raw_sensor_frame, interp_img, bin_mask, all_blobs_info, tracked_objs, final_img, stats):
    """
    Creates a single, comprehensive 2x3 grid dashboard for the tracking algorithm.
    This version is designed to fit standard screen resolutions.
    """
    # --- 1. Configuration & Canvas Setup ---
    SCALE = 3.5  # Adjusted scale to fit common screen sizes
    IMG_H, IMG_W = interp_img.shape
    SECTION_W = int(IMG_W * SCALE)
    SECTION_H = int(IMG_H * SCALE)
    
    NUM_COLS = 3
    NUM_ROWS = 2
    PANEL_HEADER_H = 30
    FOOTER_H = 100
    COLOR_BAR_H = 20
    
    CANVAS_W = SECTION_W * NUM_COLS
    CANVAS_H = (SECTION_H + PANEL_HEADER_H) * NUM_ROWS + FOOTER_H
    
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    def put_centered_text(img, text, y, font_scale, color, thickness):
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        cv2.putText(img, text, (text_x, y), font, font_scale, color, thickness)

    # --- 2. Generate All Panel Visualizations ---

    # Panel 1: Raw Sensor Data
    viz_a = normalize_with_baseline(raw_sensor_frame, stats['baseline_temp'])
    viz_a = cv2.applyColorMap(viz_a, custom_cmap1)
    viz_a = cv2.resize(viz_a, (SECTION_W, SECTION_H), interpolation=cv2.INTER_NEAREST)

    # Panel 2: Interpolated Thermal
    viz_b = normalize_with_baseline(interp_img, stats['baseline_temp'])
    viz_b = cv2.applyColorMap(viz_b, custom_cmap1)
    viz_b = cv2.resize(viz_b, (SECTION_W, SECTION_H), interpolation=cv2.INTER_NEAREST)
    base_viz_img = viz_b.copy()

    # Panel 3: Threshold Mask
    viz_c = cv2.cvtColor(bin_mask * 255, cv2.COLOR_GRAY2BGR)
    viz_c = cv2.resize(viz_c, (SECTION_W, SECTION_H), interpolation=cv2.INTER_NEAREST)

    # Panel 4: Blob Features
    viz_d = base_viz_img.copy()
    for blob_info in all_blobs_info:
        mask_disp = cv2.resize((blob_info['blob_mask'] * 255).astype(np.uint8), (SECTION_W, SECTION_H), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz_d, contours, -1, (255, 255, 255), 1)
        if blob_info['central_points'] is not None and blob_info['central_points'].size > 0:
            for y, x in blob_info['central_points']:
                cv2.circle(viz_d, (int(x * SCALE), int(y * SCALE)), 3, (0, 255, 255), -1) # Yellow
        if blob_info['centroid'] is not None:
            y, x = blob_info['centroid']
            cv2.circle(viz_d, (int(x * SCALE), int(y * SCALE)), 5, (0, 255, 0), -1) # Green
            
    # Panel 5: Tracking
    viz_e = viz_d.copy()
    for obj in tracked_objs:
        for i in range(1, len(obj.path)):
            pt1 = (int(obj.path[i-1][1] * SCALE), int(obj.path[i-1][0] * SCALE))
            pt2 = (int(obj.path[i][1] * SCALE), int(obj.path[i][0] * SCALE))
            cv2.line(viz_e, pt1, pt2, (255, 0, 0), 2) # Blue path

    # Panel 6: Final Output
    viz_f = normalize_with_baseline(final_img, stats['baseline_temp'])
    viz_f = cv2.applyColorMap(viz_f, custom_cmap1)
    viz_f = cv2.resize(viz_f, (SECTION_W, SECTION_H), interpolation=cv2.INTER_NEAREST)
    for obj in tracked_objs:
        if id(obj) not in persistent_info: # Only draw active people
            pos = (int(obj.pos[1] * SCALE), int(obj.pos[0] * SCALE))
            cv2.circle(viz_f, pos, 5, (0, 255, 0), -1) # Green centroid
            
            # Find the corresponding blob to draw its contour
            for blob_info in all_blobs_info:
                # Use np.allclose for safe floating point comparison
                if 'centroid' in blob_info and np.allclose(np.array(blob_info['centroid']), np.array(obj.pos), atol=1e-5):
                     mask_disp = cv2.resize((blob_info['blob_mask'] * 255).astype(np.uint8), (SECTION_W, SECTION_H), interpolation=cv2.INTER_NEAREST)
                     contours, _ = cv2.findContours(mask_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                     cv2.drawContours(viz_f, contours, -1, (255, 255, 255), 1)
                     break
    
    # --- 3. Assemble the Dashboard ---
    panels = [
        (viz_a, "1: Raw Sensor Data"), (viz_b, "2: Interpolated Thermal"), (viz_c, "3: Threshold Mask"),
        (viz_d, "4: Blob Features"), (viz_e, "5: Tracking"), (viz_f, "6: Final Output")
    ]
    
    for i, (panel_img, title) in enumerate(panels):
        row, col = i // NUM_COLS, i % NUM_COLS
        x_start, y_start = col * SECTION_W, row * (SECTION_H + PANEL_HEADER_H)
        
        # Area for the title
        title_area = canvas[y_start : y_start + PANEL_HEADER_H, x_start : x_start + SECTION_W]
        put_centered_text(title_area, title, PANEL_HEADER_H - 10, 0.6, (255, 255, 255), 1)
        
        # Area for the panel image
        canvas[y_start + PANEL_HEADER_H : y_start + PANEL_HEADER_H + SECTION_H, x_start : x_start + SECTION_W] = panel_img

    # --- 4. Draw Grid Lines ---
    for i in range(1, NUM_COLS):
        cv2.line(canvas, (SECTION_W * i, 0), (SECTION_W * i, (SECTION_H + PANEL_HEADER_H) * NUM_ROWS), (128, 128, 128), 1)
    for i in range(1, NUM_ROWS):
        y = i * (SECTION_H + PANEL_HEADER_H)
        cv2.line(canvas, (0, y), (CANVAS_W, y), (128, 128, 128), 1)
        
    # --- 5. Draw Footer ---
    footer_base_y = (SECTION_H + PANEL_HEADER_H) * NUM_ROWS
    cv2.putText(canvas, f"Current People: {stats['current_people']}", (20, footer_base_y + 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, f"Static Objects: {stats['persistent_objects']}", (20, footer_base_y + 60), font, 0.7, (0, 255, 255), 2)
    
    diag_x = CANVAS_W - 220
    cv2.putText(canvas, f"DHT Temp: {stats['dht_temp']:.1f} C", (diag_x, footer_base_y + 25), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f"Max Scene Temp: {stats['max_temp']:.1f} C", (diag_x, footer_base_y + 45), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f"Processing FPS: {stats['fps']:.1f}", (diag_x, footer_base_y + 65), font, 0.5, (255, 255, 255), 1)
    
    colorbar_w = 400
    colorbar_x = (CANVAS_W - colorbar_w) // 2
    colorbar_y = CANVAS_H - COLOR_BAR_H - 10
    
    gradient = np.arange(0, 256, dtype=np.uint8).reshape(1, 256, 1)
    colorbar_img = cv2.applyColorMap(gradient, custom_cmap1)
    colorbar_img = cv2.resize(colorbar_img, (colorbar_w, COLOR_BAR_H))
    canvas[colorbar_y : colorbar_y + COLOR_BAR_H, colorbar_x : colorbar_x + colorbar_w] = colorbar_img
    
    cv2.putText(canvas, f"{stats['baseline_temp']:.1f}C", (colorbar_x - 45, colorbar_y + 15), font, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, f"{stats['max_temp']:.1f}C", (colorbar_x + colorbar_w + 5, colorbar_y + 15), font, 0.5, (255, 255, 255), 1)
    put_centered_text(canvas, "Temperature Legend", colorbar_y - 8, 0.5, (255,255,255), 1)

    return canvas
########################################
# 9. Persistent Object Mask Management Functions
########################################

OBJECT_STATIC_FRAME_THRESHOLD = 100
MAX_MOVEMENT_THRESHOLD = 0.5
MIN_MOVEMENT_FOR_RESET = 2.0
DILATION_PIXELS = 2
DRIFT_WINDOW_FRAMES = 15
persistent_info = {}
static_counters = {} 

def compute_locked_mask(centroid, size, dilation_pixels=DILATION_PIXELS):
    mask = np.zeros((71, 71), dtype=np.uint8)
    r = int(math.sqrt(size / math.pi)) + dilation_pixels
    center = (int(round(centroid[1])), int(round(centroid[0])))
    cv2.circle(mask, center, r, 1, thickness=-1)
    return mask

def update_persistent_objects(tracked_objects_list):
    global persistent_info, static_counters
    current_persistent_ids = set(persistent_info.keys())
    active_track_ids = {id(obj) for obj in tracked_objects_list}

    # Remove info for objects that are no longer tracked
    ids_to_remove = current_persistent_ids - active_track_ids
    for obj_id_remove in ids_to_remove:
        del persistent_info[obj_id_remove]
    
    # Remove counters for inactive tracks
    counters_to_remove = set(static_counters.keys()) - active_track_ids
    for c_id in counters_to_remove:
        del static_counters[c_id]

    for obj in tracked_objects_list:
        obj_id = id(obj)
        if obj_id in persistent_info:
            # If already persistent, check if it has moved significantly to reset
            locked_centroid = persistent_info[obj_id]["locked_centroid"]
            dist = math.sqrt((obj.pos[0] - locked_centroid[0])**2 + (obj.pos[1] - locked_centroid[1])**2)
            if dist > MIN_MOVEMENT_FOR_RESET:
                del persistent_info[obj_id]
                if obj_id in static_counters:
                    del static_counters[obj_id]
        else:
            # Check if object has enough history for drift comparison
            if len(obj.path) >= DRIFT_WINDOW_FRAMES:
                # Compare current position to DRIFT_WINDOW_FRAMES ago
                past_pos = obj.path[-DRIFT_WINDOW_FRAMES]
                dist = math.sqrt((obj.pos[0] - past_pos[0])**2 + (obj.pos[1] - past_pos[1])**2)
                
                if dist < MAX_MOVEMENT_THRESHOLD:
                    # Increment consecutive static counter
                    static_counters[obj_id] = static_counters.get(obj_id, 0) + 1
                    
                    # If reached threshold, mark as persistent
                    if static_counters[obj_id] >= OBJECT_STATIC_FRAME_THRESHOLD:
                        locked_mask = compute_locked_mask(obj.pos, obj.size)
                        persistent_info[obj_id] = {
                            "locked_centroid": obj.pos,
                            "locked_mask": locked_mask
                        }
                        del static_counters[obj_id]  # Clean up counter
                else:
                    # Movement detected, reset counter
                    static_counters[obj_id] = 0
            else:
                # Not enough history yet
                static_counters[obj_id] = 0

def generate_combined_persistent_mask():
    combined_mask = np.zeros((71, 71), dtype=np.uint8)
    for obj_id in list(persistent_info.keys()):
        if obj_id in persistent_info:
             combined_mask = cv2.bitwise_or(combined_mask, persistent_info[obj_id]["locked_mask"])
    return combined_mask

########################################
# 8. Main Loop: Processing Sensor Data
########################################

def main():
    global tracked_objects, latest_thermal_data, MATCH_THRESHOLD_CENTROID, CENTRAL_POINT_DIST_THRESHOLD, JACCARD_THRESHOLD_CP
    
    cv2.namedWindow("Occupancy Monitor Visualization", cv2.WINDOW_AUTOSIZE)
    
    frame_count, fps = 0, 0.0
    start_time = time.time()

    while True:
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count, start_time = 0, time.time()

        sensor_frame = latest_thermal_data.copy()
        interp_img = interpolate8to71(sensor_frame)
        baseline_temp = latest_dht_temp - 8
        bin_mask = threshold_image(interp_img, latest_dht_temp)
        num_labels, labels = connected_components(bin_mask)
        
        current_detected_blobs_info = []
        for label in range(1, num_labels):
            res = compute_blob_features(labels, label, interp_img.shape)
            if res is None: continue
            blob_mask_val, size_val, centroid_val, central_points_val = res
            current_detected_blobs_info.append({ 'blob_mask': blob_mask_val, 'size': size_val, 'centroid': tuple(centroid_val), 'central_points': central_points_val })
        
        for obj in tracked_objects: obj.updated = False
        
        unmatched_blobs = list(range(len(current_detected_blobs_info)))

        for i, obj in enumerate(tracked_objects):
            if not unmatched_blobs: break
            best_blob_idx, best_distance_centroid = -1, float('inf')
            for blob_list_idx, current_blob_orig_idx in enumerate(unmatched_blobs):
                blob_info = current_detected_blobs_info[current_blob_orig_idx]
                distance = abs(blob_info['centroid'][0] - obj.pos[0]) + abs(blob_info['centroid'][1] - obj.pos[1])
                if distance < best_distance_centroid and distance < MATCH_THRESHOLD_CENTROID:
                    best_distance_centroid, best_blob_idx = distance, blob_list_idx
            if best_blob_idx != -1:
                matched_blob_orig_idx = unmatched_blobs.pop(best_blob_idx)
                matched_blob_info = current_detected_blobs_info[matched_blob_orig_idx]
                obj.update(matched_blob_info['centroid'], matched_blob_info['size'], matched_blob_info['central_points'])

        remaining_tracked_objects_indices = [i for i, obj in enumerate(tracked_objects) if not obj.updated]
        temp_unmatched_blobs_info = [current_detected_blobs_info[i] for i in unmatched_blobs]
        for track_idx in remaining_tracked_objects_indices:
            obj = tracked_objects[track_idx]
            best_jaccard_blob_info, best_jaccard_score = None, 0.0
            for blob_info_item in temp_unmatched_blobs_info:
                obj_cp = obj.central_points if obj.central_points.size > 0 else np.array([])
                blob_cp = blob_info_item['central_points'] if blob_info_item['central_points'].size > 0 else np.array([])
                
                if obj_cp.ndim == 1 and obj_cp.size > 0: obj_cp = obj_cp.reshape(-1, 2)
                if blob_cp.ndim == 1 and blob_cp.size > 0: blob_cp = blob_cp.reshape(-1, 2)
                j_score = calculate_jaccard_index_cp(obj_cp, blob_cp, CENTRAL_POINT_DIST_THRESHOLD)
                if j_score > best_jaccard_score and j_score >= JACCARD_THRESHOLD_CP:
                    best_jaccard_score, best_jaccard_blob_info = j_score, blob_info_item
            if best_jaccard_blob_info is not None:
                obj.update(best_jaccard_blob_info['centroid'], best_jaccard_blob_info['size'], best_jaccard_blob_info['central_points'])
                temp_unmatched_blobs_info = [b for b in temp_unmatched_blobs_info if b is not best_jaccard_blob_info]

        current_blob_centroids_for_new_tracks = {tuple(b['centroid']) for b in temp_unmatched_blobs_info}
        for orig_idx in unmatched_blobs:
            if tuple(current_detected_blobs_info[orig_idx]['centroid']) in current_blob_centroids_for_new_tracks:
                blob_info = current_detected_blobs_info[orig_idx]
                new_obj = HumanObject(label=0, pos=blob_info['centroid'], size=blob_info['size'], central_points=blob_info['central_points'])
                tracked_objects.append(new_obj)
        
        new_tracked_objects_list = [obj for obj in tracked_objects if obj.updated or obj.virtual_age < 3]
        for obj in tracked_objects:
            if not obj.updated: obj.virtual_propagate()
        tracked_objects[:] = new_tracked_objects_list
        
        update_persistent_objects(tracked_objects)
        persistent_mask = generate_combined_persistent_mask()
        final_img = np.copy(interp_img)
        final_img[persistent_mask == 1] = baseline_temp
        
        # Calculate counts
        current_people = sum(1 for obj in tracked_objects if id(obj) not in persistent_info)
        num_persistent_objects = len(persistent_info)
        
        print(f"Frame Processed | People: {current_people} | Static Objects: {num_persistent_objects} | FPS: {fps:.1f}")

        # --- Generate and Display Dashboard ---
        stats = {
            'current_people': current_people,
            'persistent_objects': num_persistent_objects,
            'dht_temp': latest_dht_temp,
            'max_temp': interp_img.max(),
            'fps': fps,
            'baseline_temp': baseline_temp
        }
        
        dashboard_canvas = create_combined_visualization(
            sensor_frame, interp_img, bin_mask, current_detected_blobs_info, tracked_objects, final_img, stats
        )
        cv2.imshow("Occupancy Monitor Visualization", dashboard_canvas)
        
        # Data Export
        export_data(sensor_frame, latest_dht_temp, current_people, num_persistent_objects)
        
        key = cv2.waitKey(50)
        if key == 27: break

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    time.sleep(1)
    
    # --- Tunable Parameters ---
    MATCH_THRESHOLD_CENTROID = 25
    CENTRAL_POINT_DIST_THRESHOLD = 7
    JACCARD_THRESHOLD_CP = 0.25
    
    main()
    
    cv2.destroyAllWindows()
    csv_file.close()
