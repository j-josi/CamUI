# System level imports
import os, io, logging, json, time, re, glob, math, tempfile, zipfile
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from threading import Condition
import threading, subprocess
import argparse
import copy

# Flask imports
from flask import Flask, render_template, request, jsonify, Response, send_file, abort, session, redirect, url_for
import secrets

from libcamera import Transform, controls

# Image handeling imports
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps, ExifTags

from camera_manager import CameraManager
from media_gallery import MediaGallery

####################
# Initialize Flask 
####################

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generates a random 32-character hexadecimal string
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie#samesitesamesite-value
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

####################
# Initialize picamera2 
####################

# Set debug level to Warning
# Picamera2.set_logging(Picamera2.DEBUG)

##### Uncomment the line below if you want to limt the number of cameras connected (change the number to index which camera you want)
# global_cameras = [global_cameras[0]]

##### Uncomment the line below simulate having no cameras connected
# global_cameras = []

# print(f'\nInitialize picamera2 - Cameras Found:\n{global_cameras}\n')

####################
# Initialize default values 
####################

version = "2.0.0 - BETA"
project_title = "CamUI - for picamera2"
firmware_control = False

# Mediamtx
mediamtx_webrtc_port = 8889

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Set path for files storing the last/active camera configuration and camera modules information
last_config_path = os.path.join(current_dir, 'camera-last-config.json')
camera_module_info_path = os.path.join(current_dir, 'camera-module-info.json')
camera_control_db_path = os.path.join(current_dir, 'camera_controls_db.json')

# Set the path where the camera profiles are stored
camera_profile_folder = os.path.join(current_dir, 'static/camera_profiles')
app.config['camera_profile_folder'] = camera_profile_folder
# Create the folder if it does not exist
os.makedirs(app.config['camera_profile_folder'], exist_ok=True)

# Set the path where the images and videos will be stored for the media gallery
media_upload_folder = os.path.join(current_dir, 'static/gallery')
app.config['media_upload_folder'] = media_upload_folder
# Create the folder if it does not exist
os.makedirs(app.config['media_upload_folder'], exist_ok=True)

# For the media gallery set items per page
# items_per_page = 12

# Define the minimum required configuration
minimum_last_config = {
    "cameras": []
}

# Set epoch and (monotonic) start timestamp
DEFAULT_EPOCH = datetime(1970, 1, 1)
_MONOTONIC_START = time.monotonic()

####################
# Initialize CameraManager
####################

camera_manager = CameraManager(
    camera_module_info_path=camera_module_info_path,
    last_config_path=last_config_path,
    media_upload_folder=media_upload_folder,
    camera_control_db_path=camera_control_db_path
)

camera_manager.init_cameras()

# camera_manager.detect_connected_cameras()
# camera_manager.sync_last_config()
# camera_manager.init_connected_cameras()


####################
# Initialize Media Gallery 
####################
media_gallery_manager = MediaGallery(media_upload_folder)

# Function to load or initialize configuration
def load_or_initialize_config(file_path, default_config):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                config = json.load(file)
                if not config:  # Check if the file is empty
                    raise ValueError("Empty configuration file")
            except (json.JSONDecodeError, ValueError):
                # If file is empty or invalid, create new config
                with open(file_path, 'w') as file:
                    json.dump(default_config, file, indent=4)
                config = default_config
    else:
        # Create the file with minimum configuration if it doesn't exist
        with open(file_path, 'w') as file:
            json.dump(default_config, file, indent=4)
        config = default_config
    return config

def get_active_profile():
    return load_or_initialize_config(last_config_path, minimum_last_config)

def get_profiles():
    profiles = []
    if not os.path.exists(camera_profile_folder):
        os.makedirs(camera_profile_folder)

    for f in os.listdir(camera_profile_folder):
        if f.endswith(".json"):
            path = os.path.join(camera_profile_folder, f)
            try:
                with open(path, "r") as pf:
                    data = json.load(pf)
                profiles.append({
                    "filename": f,
                    "model": data.get("model", "Unknown"),
                    "active": (f == get_active_profile()["cameras"][0]["Config_Location"])
                })
            except Exception as e:
                print(f"Error loading {f}: {e}")
    return profiles

# def control_template():
#     with open(os.path.join(current_dir, "camera_controls_db.json"), "r") as f:
#         settings = json.load(f)
#     return settings

# def get_camera_info(camera_model, camera_module_info):
#     return next(
#         (module for module in camera_module_info["camera_modules"] if module["sensor_model"] == camera_model),
#         next(module for module in camera_module_info["camera_modules"] if module["sensor_model"] == "Unknown")
#     )

def system_time_is_synced() -> bool:
    """Check if system time of raspberrypi is synced with NTP server"""
    try:
        result = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True
        )
        return result.stdout.strip().lower() == "yes"
    except Exception:
        return False

def generate_filename(cam_num: int, file_extension: str = ".jpg") -> str:
    # Normalize file extension
    if not file_extension.startswith("."):
        file_extension = "." + file_extension

    if system_time_is_synced():
        ts = datetime.now()
    else:
        elapsed = int(time.monotonic() - _MONOTONIC_START)
        ts = DEFAULT_EPOCH + timedelta(seconds=elapsed)

    timestamp = ts.strftime("%Y-%m-%d_%H-%M-%S")

    if cam_num == 0:
        return f"{timestamp}{file_extension}"
    else:
        return f"{timestamp}_cam{cam_num}{file_extension}"

####################
# Flask routes - WebUI routes 
####################

@app.context_processor
def inject_theme():
    theme = session.get('theme', 'light')  # Default to 'light'
    return dict(version=version, title=project_title, theme=theme)

@app.context_processor
def inject_camera_list():
    camera_list = [
        (camera.camera_info, camera.get_camera_module_spec())
        for camera in camera_manager.cameras.values()  # <-- .values() für CameraObject-Instanzen
    ]
    return dict(camera_list=camera_list, navbar=True)

@app.route('/set_theme/<theme>')
def set_theme(theme):
    session['theme'] = theme
    return jsonify(success=True, ok=True, message="Theme updated successfully")

# Define 'home' route
@app.route('/')
def home():
    camera_list = [
        (camera.camera_info, camera.get_camera_module_spec())
        for camera in camera_manager.cameras.values()  # <-- .values() für CameraObject-Instanzen
    ]

    return render_template('home.html', active_page='home')

@app.route('/camera_info_<int:camera_num>')
def camera_info(camera_num):
    # Check if the camera number exists
    camera = camera_manager.get_camera(camera_num)
    if not camera:
        return render_template('error.html', message="Error: Camera not found"), 404
    # Get camera module spec
    camera_module_spec = camera.camera_module_spec

    return render_template('camera_info.html', camera_data=camera_module_spec, camera_num=camera_num)

# long polling -> return instantly when state of camera.active_recording changes, otherwise return after 15s
@app.route("/camera_status_long/<int:camera_num>")
def camera_status_long(camera_num):
    try:
        camera = camera_manager.get_camera(camera_num)
        if not camera:
            return jsonify(success=False, error="Camera not found"), 404

        last_state = request.args.get("state", "false") == "true"
        timeout = 15
        start = time.time()

        while time.time() - start < timeout:
            if camera.active_recording != last_state:
                return jsonify(
                    success=True,
                    active_recording=camera.active_recording
                )
            time.sleep(0.2)

        return jsonify(
            success=True,
            active_recording=camera.active_recording
        )

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route("/about")
def about():
    return render_template("about.html", active_page='about')

@app.route('/system_settings')
def system_settings():
    # Load camera module info
    print(camera_manager.camera_module_info)
    return render_template('system_settings.html', firmware_control=firmware_control, camera_modules=camera_manager.camera_module_info.get("camera_modules", []))

@app.route('/set_camera_config', methods=['POST'])
def set_camera_config():
    data = request.get_json()
    sensor_model = data.get('sensor_model')
    config_path = "/boot/firmware/config.txt"

    try:
        with open(config_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        modified = False
        found_anchor = False
        i = 0

        while i < len(lines):
            line = lines[i]

            if "# Automatically load overlays for detected cameras" in line:
                found_anchor = True
                new_lines.append(line)
                i += 1

                # Look for camera_auto_detect line
                if i < len(lines) and lines[i].strip().startswith("camera_auto_detect="):
                    # Replace with 0
                    new_lines.append("camera_auto_detect=0\n")
                    i += 1
                else:
                    # Add camera_auto_detect=0 if missing
                    new_lines.append("camera_auto_detect=0\n")

                # Check for dtoverlay line
                if i < len(lines) and lines[i].strip().startswith("dtoverlay="):
                    # Replace this one line only
                    new_lines.append(f"dtoverlay={sensor_model}\n")
                    i += 1
                else:
                    # Insert dtoverlay line
                    new_lines.append(f"dtoverlay={sensor_model}\n")

                modified = True
                continue

            new_lines.append(line)
            i += 1

        if not found_anchor:
            return jsonify({"message": "Anchor section not found in config.txt"}), 400

        # Write to temp file
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.writelines(new_lines)
            tmp_path = tmp.name

        # Move into place with sudo
        result = subprocess.run(["sudo", "mv", tmp_path, config_path], capture_output=True)

        if result.returncode != 0:
            return jsonify({"message": f"Error writing config: {result.stderr.decode()}"}), 500

        return jsonify({"message": f"Camera '{sensor_model}' set in boot config!"})

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/reset_camera_detection', methods=['POST'])
def reset_camera_detection():
    config_path = "/boot/firmware/config.txt"
    try:
        with open(config_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() == "camera_auto_detect=0":
                new_lines.append("camera_auto_detect=1\n")
                # Check if the next line is a dtoverlay
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("dtoverlay="):
                    i += 2  # skip both lines
                    continue
                else:
                    i += 1
                    continue
            else:
                new_lines.append(line)
                i += 1

        # Write to temp file
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.writelines(new_lines)
            tmp_path = tmp.name

        # Move into place with sudo
        result = subprocess.run(["sudo", "mv", tmp_path, config_path], capture_output=True)
       
        if result.returncode != 0:
            return jsonify({"message": f"Error writing config: {result.stderr.decode()}"}), 500

        return jsonify({"message": f"Camera detection reset to automatic."})

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    try:
        subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=True)
        return jsonify({"message": "System is shutting down."})
    except subprocess.CalledProcessError as e:
        return jsonify({"message": f"Error: {e}"}), 500

@app.route('/restart', methods=['POST'])
def restart():
    try:
        subprocess.run(['sudo', 'reboot'], check=True)
        return jsonify({"message": "System is restarting."})
    except subprocess.CalledProcessError as e:
        return jsonify({"message": f"Error: {e}"}), 500

####################
# Flask routes - Camera Control 
####################

@app.route("/camera_mobile_<int:camera_num>")
def camera_mobile(camera_num):
    feature = "camera mobile view"
    return render_template('coming_soon.html', feature=feature)

# @app.route("/camera_mobile_<int:camera_num>")
# def camera_mobile(camera_num):
#     try:
#         camera = camera_manager.get_camera(camera_num)
#         if not camera:
#             return render_template('camera_not_found.html', camera_num=camera_num)
#         # Get camera settings
#         live_controls = camera.live_controls
#         print(live_controls)
#         sensor_modes = camera.sensor_modes
#         active_mode_index = camera.get_sensor_mode()
#         # Find the last image taken by this specific camera
#         last_image = None
#         last_image = media_gallery_manager.find_last_image_taken()
#         return render_template('camera_mobile.html', camera=camera.camera_info, settings=live_controls, sensor_modes=sensor_modes, active_mode_index=active_mode_index, last_image=last_image, profiles=get_profiles(),navbar=False, theme='dark', mode="mobile") 
#     except Exception as e:
#         logging.error(f"Error loading camera view: {e}")
#         return render_template('error.html', error=str(e))

@app.route("/camera_<int:camera_num>")
def camera(camera_num):
    try:
        camera = camera_manager.get_camera(camera_num)
        if not camera:
            return render_template('camera_not_found.html', camera_num=camera_num)

        # Live Controls (alle Settings aus camera_controls_db.json)
        live_controls = camera.live_controls

        # Letztes Bild (optional)
        last_image = media_gallery_manager.find_last_image_taken()

        return render_template(
            'camera.html',
            camera=camera.camera_info,
            settings=live_controls,
            last_image=last_image,
            profiles=get_profiles(),
            mode="desktop"
        )
    except Exception as e:
        logging.error(f"Error loading camera view: {e}")
        return render_template('error.html', error=str(e))

@app.route("/capture_still_<int:camera_num>", methods=["POST"])
def capture_still(camera_num):
    try:
        logging.debug(f"📸 Received capture request for camera {camera_num}")

        camera = camera_manager.get_camera(camera_num)
        if not camera:
            logging.warning(f"❌ Camera {camera_num} not found.")
            return jsonify(success=False, message="Camera not found"), 404

        # Generate the new filename
        timestamp = int(time.time())  # Current Unix timestamp
        if len(camera_manager.list_cameras()) == 1:
            image_filename = f"{timestamp}.jpg"
        else:
            image_filename = f"{timestamp}_cam{camera_num}.jpg"

        logging.debug(f"📁 New image filename: {image_filename}")

        image_filepath = os.path.join(app.config['media_upload_folder'], image_filename)
        image_filepath = camera.capture_still(image_filepath, camera.camera_profile["saveRAW"])

        camera.reconfigure_video_pipeline()
        camera.start_streaming()

        if image_filepath:
            return jsonify(success=True, message="Still image captured successfully", image=image_filepath)
        else:
            return jsonify(success=False, message="Failed to capture still image", image=image_filepath)

    except Exception as e:
        logging.error(f"🔥 Error capturing still image: {e}")
        camera.picam2.stop()
        time.sleep(0.5)
        camera.reconfigure_video_pipeline()
        camera.start_streaming()

        return jsonify(success=False, message=str(e)), 500

@app.route('/snapshot_<int:camera_num>')
def snapshot(camera_num):
    camera = camera_manager.get_camera(camera_num)
    if camera:
        image_filename = f"snapshot_{camera_num}.jpg"
        image_filepath = os.path.join(app.config['media_upload_folder'], filename)

        filepath = camera.capture_still_from_feed(image_filepath)
        
        if image_filepath:
            time.sleep(1)  # Ensure the image is saved
            return send_file(image_filepath, as_attachment=False, download_name="snapshot.jpg", mimetype='image/jpeg')
    else:
        abort(404)

# Optional: Default /video_feed_0 → /cam/
# @app.route('/video_feed_<int:camera_num>')
# def video_feed_default(camera_num):
#     return video_feed_proxy(camera_num, "cam")

@app.route('/video_feed_<int:camera_num>')
def video_feed(camera_num):
    camera = camera_manager.get_camera(camera_num)
    if camera is None:
        abort(404)

    # Get ip address of this raspberrypi
    host_ip = request.host.split(":")[0]
    
    return redirect(f"http://{host_ip}:{mediamtx_webrtc_port}/cam{camera_num}/", code=302)

@app.route("/video_webrtc_url/<int:camera_num>")
def video_webrtc_url(camera_num):
    camera = camera_manager.get_camera(camera_num)
    if camera is None:
        abort(404)

    # Get ip address of this raspberrypi
    host_ip = request.host.split(":")[0]
    
    return jsonify({
        "url": f"http://{host_ip}:{mediamtx_webrtc_port}/cam{camera_num}/whep"
    })

@app.route("/start_recording/<int:camera_num>")
def start_recording(camera_num):
    camera = camera_manager.get_camera(camera_num)
    if not camera:
        return jsonify(success=False, error="Invalid camera number"), 400

    # Generate the new filename
    timestamp = int(time.time())  # Current Unix timestamp
    filename_recording = f"{timestamp}_cam_{camera_num}.mp4"
    logging.debug(f"📁 New video filename: {filename_recording}")

    success = camera.start_recording(filename_recording)
    if success:
        message = f"Recording of file {filename_recording} started successfully"
    else:
        message = "Failed to start recording"

    return jsonify(success=success, message=message)

@app.route("/stop_recording/<int:camera_num>")
def stop_recording(camera_num):
    camera = camera_manager.get_camera(camera_num)
    if not camera:
        return jsonify(success=False, error="Invalid camera number"), 400

    success = camera.stop_recording()
    if success:
        message = f"Recording of file {camera.filename_recording} stopped successfully"
    else:
        message = "Failed to stop recording"

    return jsonify(success=success, message=message)

@app.route('/preview_<int:camera_num>', methods=['POST'])
def preview(camera_num):
    try:
        camera = camera_manager.get_camera(camera_num)
        if camera:
            filepath = f'snapshot/pimage_preview_{camera_num}'
            preview_path = camera.capture_still(filepath)
            return jsonify(success=True, message="Photo captured successfully", image_path=preview_path)
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/update_setting', methods=['POST'])
def update_setting():
    try:
        data = request.json  # Get JSON data from the request
        camera_num = data.get("camera_num")  # New field for camera selection
        setting_id = data.get("id")
        new_value = data.get("value")
        # Debugging: Print the received values
        print(f"Received update for Camera {camera_num}: {setting_id} -> {new_value}")
        camera = camera_manager.get_camera(camera_num)
        camera.update_settings(setting_id, new_value)
        # ✅ At this stage, we're just verifying the data. No changes to the camera yet.
        return jsonify({
            "success": True,
            "message": f"Received setting update for Camera {camera_num}: {setting_id} -> {new_value}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/camera_controls')
def redirect_to_home():
    return redirect(url_for('home'))

@app.route("/set_recording_resolution", methods=["POST"])
def set_recording_resolution():
    data = request.get_json()
    cam = cameras[data["camera_num"]]
    cam.set_recording_resolution((data["width"], data["height"]))
    return jsonify({"status": "ok"})

@app.route("/set_streaming_resolution", methods=["POST"])
def set_streaming_resolution():
    data = request.get_json()
    cam = cameras[data["camera_num"]]
    cam.set_streaming_resolution((data["width"], data["height"]))
    return jsonify({"status": "ok"})

####################
# Camera Profile routes 
####################

@app.route("/get_camera_profile", methods=["GET"])
def get_camera_profile():
    camera_num = request.args.get("camera_num", type=int)
    camera = camera_manager.get_camera(camera_num)
    if camera:
        camera_profile = camera.camera_profile  # Fetch current controls
        return jsonify(success=True, camera_profile=camera_profile)
    else:
        return jsonify(success=False, camera_profile="")

@app.route('/create_profile_<int:camera_num>', methods=['POST'])
def create_profile(camera_num):
    data = request.json
    filename = data.get("filename")

    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    camera = camera_manager.get_camera(camera_num)
    success = camera.create_profile(filename)

    if success:
        return jsonify({"message": f"Profile '{filename}' created successfully"}), 200
    else:
        return jsonify({"error": "Failed to save profile"}), 500

@app.route("/reset_profile_<int:camera_num>", methods=["POST"])
def reset_profile(camera_num):
    camera = camera_manager.get_camera(camera_num)
    if camera and camera.reset_to_default():
        return jsonify({"success": True, "message": "Profile reset to default values"})
    else:
        return jsonify({"success": False, "message": "Failed to reset profile to default values"}), 404

@app.route("/delete_profile_<int:camera_num>", methods=["POST"])
def delete_profile(camera_num):
    data = request.get_json()
    filename = data.get("filename")

    if not filename:
        return jsonify({"success": False, "message": "No filename provided"}), 400

    profile_path = os.path.join(camera_profile_folder, filename)
    if not os.path.exists(profile_path):
        return jsonify({"success": False, "message": "Profile not found"}), 404

    try:
        os.remove(profile_path)
        return jsonify({"success": True, "message": "Profile deleted"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/fetch_metadata_<int:camera_num>")
def fetch_metadata(camera_num):
    camera = camera_manager.get_camera(camera_num)

    if not camera:
        return jsonify({"error": "Invalid camera number"}), 400

    metadata = camera.capture_metadata()  # Get metadata for the selected camera
    print(f"Camera {camera_num} Metadata: {metadata}")  # Log metadata
    return jsonify(metadata)  # Return as JSON

@app.route("/load_profile", methods=["POST"])
def load_profile():
    data = request.get_json()
    profile_name = data.get("profile_name")
    camera_num = data.get("camera_num")

    if not profile_name:
        return jsonify({"error": "Profile name is missing"}), 400
    if camera_num is None:
        return jsonify({"error": "Camera number is missing"}), 400

    camera = camera_manager.get_camera(camera_num)
    if camera:
        success = camera.load_camera_profile(profile_name)
        if success:
            return jsonify({"message": f"Profile '{profile_name}' loaded successfully"})
        else:
            return jsonify({"error": "Failed to load profile"}), 500
    else:
        return jsonify({"error": "Invalid camera number"}), 400

@app.route("/get_profiles")
def _get_profiles():
    # return list_profiles()
    return get_profiles()

####################
# Flask routes - Media gallery routes 
####################

@app.route('/media_gallery')
def media_gallery():
    media_type = request.args.get('type', 'all')
    return render_template(
        'media_gallery.html',
        media_type=media_type,
        active_page='media_gallery'
    )

@app.route('/get_media_slice')
def get_media_slice():
    """AJAX route for endless scroll."""
    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', 20, type=int)
    media_type = request.args.get('type', 'all')

    media_files = media_gallery_manager.get_media_slice(offset=offset, limit=limit, type=media_type)

    response = {
        'media_files': media_files
    }

    return jsonify(response)

@app.route('/view_image/<filename>')
def view_image(filename):
    return render_template('view_image.html', filename=filename)

@app.route('/delete_media/<filename>', methods=['DELETE'])
def delete_media(filename):
    success, message = media_gallery_manager.delete_media(filename)

    if success:
        return jsonify({"success": True, "message": message}), 200
    else:
        return jsonify({"success": False, "message": message}), 404 if "not found" in message else 500

@app.route('/image_edit/<filename>')
def edit_image(filename):
    return render_template('image_edit.html', filename=filename)

@app.route("/apply_filters", methods=["POST"])
def apply_filters():
    try:
        filename = request.form["filename"]
        brightness = float(request.form.get("brightness", 1.0))
        contrast = float(request.form.get("contrast", 1.0))
        rotation = float(request.form.get("rotation", 0))

        # Vollständigen Pfad der Datei
        img_path = os.path.join(app.config['media_upload_folder'], filename)

        # Filter anwenden
        edited_filepath = media_gallery.apply_filter(
            img_path,
            brightness=brightness,
            contrast=contrast,
            rotation=rotation
        )

        if edited_filepath:
            # Datei zurückgeben
            edited_filename = os.path.basename(edited_filepath)
            return send_from_directory(app.config['media_upload_folder'], edited_filename)
        else:
            return jsonify(success=False, message="Failed to apply filters"), 500

    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/download_image/<filename>', methods=['GET'])
def download_image(filename):
    try:
        image_path = os.path.join(app.config['media_upload_folder'], filename)
        return send_file(image_path, as_attachment=True)
    except Exception as e:
        print(f"\nError downloading image:\n{e}\n")
        abort(500)

@app.route("/download_media_bulk", methods=["POST"])
def download_media_bulk():
    files_json = request.form.get("files", "[]")
    files = json.loads(files_json)
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            path = os.path.join(app.config["media_upload_folder"], f)
            if os.path.exists(path):
                zf.write(path, arcname=f)

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name="media_selection.zip"
    )

@app.route('/save_edit', methods=['POST'])
def save_edit():
    try:
        data = request.json
        filename = data.get('filename')
        edits = data.get('edits', {})
        save_option = data.get('saveOption')
        new_filename = data.get('newFilename')

        success, message = media_gallery_manager.save_edit(filename, edits, save_option, new_filename)

        return jsonify({'success': success, 'message': message})

    except Exception as e:
        logging.error(f"Error in save_edit route: {e}")
        return jsonify({'success': False, 'message': 'Error saving edit'}), 500

####################
# Flask routes - Misc Routes
####################

@app.route('/beta')
def beta():
    return render_template('beta.html')

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

####################
# Start Flask 
####################

if __name__ == "__main__":
    # Parse any argument passed from command line
    parser = argparse.ArgumentParser(description='PiCamera2 WebUI')
    parser.add_argument('--port', type=int, default=8080, help='Port number to run the web server on')
    parser.add_argument('--ip', type=str, default='0.0.0.0', help='IP to which the web server is bound to')
    args = parser.parse_args()
    # If there are no arguments the port will be 8080 and ip 0.0.0.0
    # uncoment following line, if no external WSGI server (i.e. Gunicorn) is used and Flask should use its internal server (not recommended for production systems)
    # app.run(host=args.ip, port=args.port)
