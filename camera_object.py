# System level imports
import os, io, logging, json, time, re, glob, math, tempfile, zipfile
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from threading import Condition
import threading, subprocess
import argparse
import copy

# picamera2 imports
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, PyavOutput, FfmpegOutput
from libcamera import Transform, controls

# Image handeling imports
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps, ExifTags



# temp
current_dir = os.path.dirname(os.path.abspath(__file__))
camera_profile_folder = os.path.join(current_dir, 'static/camera_profiles')
# Max. supported resolution of h264 encoder
ENCODER_MAX = (1920, 1080)


####################
# CameraObject that will store the itteration of 1 or more cameras
####################

class CameraObject:
    def __init__(self, camera, camera_module_info, upload_folder, last_config_path, camera_control_db_path):
        self.camera_info = camera
        self.camera_module_info = camera_module_info
        self.upload_folder = upload_folder
        self.last_config_path = last_config_path
        self.camera_control_db_path = camera_control_db_path
        self.lock = threading.Lock()
        self.camera_num = self.camera_info['Num']
        # Generate default Camera profile
        self.camera_profile = self.generate_camera_profile()
        # Init camera to picamera2 using the camera number
        self.picam2 = Picamera2(self.camera_num)
        # Get Camera specs
        self.camera_module_spec = self.get_camera_module_spec()
        # Fetch Avaialble Sensor modes and generate available resolutions
        self.sensor_modes = self.picam2.sensor_modes
        self.camera_resolutions = self.generate_camera_resolutions()
        # Get all available video resolutions for this camera/sensor
        self.video_resolutions = self.generate_video_resolutions()
        # Stream Encoder
        self.encoder_stream = H264Encoder(bitrate=8_000_000)
        # TODO function to detect microphone
        self.encoder_stream.audio = False
        self.encoder_stream.audio_output = {'codec_name': 'libopus'}
        self.encoder_stream.audio_sync = 0
        # Stream Output (PyavOutput processed by Mediamtx)
        self.output_stream = PyavOutput(f"rtsp://127.0.0.1:8554/cam{self.camera_num}", format="rtsp")
        # Stream activity flag
        self.active_stream = False
        # Recording Encoder
        self.encoder_recording = H264Encoder(bitrate=8_000_000)
        # self.resolution_recording = self.video_resolutions[0]
        # Recording Output
        self.output_recording = None
        self.filename_recording = None
        # Recording activity flag
        self.active_recording = False
        # Initialize configs as empty dictionaries for the still and video configs
        self.init_configure_camera()
        # Compare camera controls DB flushing out settings not avaialbe from picamera2
        self.live_controls = self.initialize_controls_template(self.get_ui_controls(), self.camera_control_db_path)

        # Set capture flag and set placeholder image
        self.active_capture_still = False
        # self.placeholder_frame = self.generate_placeholder_frame()  # Create placeholder
        # Audio
        # TODO function to detect microphone
        self.audio = False
        self.main_stream = "recording"
        self.lores_stream = "streaming"

        # Load and apply saved/active camera profile if one exists and sync meta data
        self.load_saved_camera_profile()

        # Start streaming
        self.start_streaming()

        # Final debug statements
        print(f"Available Camera Controls: {self.get_ui_controls()}")
        print(f"Available Sensor Modes: {self.sensor_modes}")
        print(f"Available Resolutions: {self.camera_resolutions}")
        print(f"Final Camera Profile: {self.camera_profile}")

    #-----
    # Camera Config Functions
    #-----

    def get_ui_controls(self):
        # copy camera controls from picamera2.camera_controls (which inherits the values from libcamera driver) and set/crop to custom/reasonable ranges used by ui slider/controls
        ui_controls = copy.deepcopy(self.picam2.camera_controls)

        if "ExposureTime" in ui_controls:
            min_exposure_time = 100 # microseconds
            max_exposure_time = 100000 # microseconds
            default_exposure_time = 500
            ui_controls["ExposureTime"] = (min_exposure_time, max_exposure_time, default_exposure_time)
        if "ColourTemperature" in ui_controls:
            min_color_temp = 100 # Kelvin
            max_color_temp = 10000 # Kelvin
            default_color_temp = None
            ui_controls["ColourTemperature"] = (min_color_temp, max_color_temp, default_color_temp)
        return ui_controls

    def init_configure_camera(self):
        self.still_config = {}
        self.video_config = {}
        self.still_config = self.picam2.create_still_configuration()
        self.video_config = self.picam2.create_video_configuration()

    def load_saved_camera_profile(self):
        """Load the saved camera config if available."""
        print("DEBUG: load_saved_camera_profile")
        if self.camera_info.get("Has_Config") and self.camera_info.get("Config_Location"):
            self.load_camera_profile(self.camera_info["Config_Location"])

    def load_camera_profile(self, profile_filename):
        """Load and apply a camera profile from a given filename."""
        profile_path = os.path.join(camera_profile_folder, profile_filename)
        if not os.path.exists(profile_path):
            print(f"Profile file not found: {profile_path}")
            return False
        try:
            with open(profile_path, "r") as f:
                profile_data = json.load(f)
            # Load the profile before applying any settings
            self.camera_profile = profile_data
            # Apply settings after loading the profile
            self.update_settings('recording_resolution', self.camera_profile['resolutions']['recording_resolution'], True)
            self.update_settings('streaming_resolution', self.camera_profile['resolutions']['streaming_resolution'], True)
            self.update_settings('StillCaptureResolution', self.camera_profile['resolutions']['StillCaptureResolution'], True)
            self.update_settings('hflip', self.camera_profile['hflip'], True)
            self.update_settings('vflip', self.camera_profile['vflip'], True)
            self.update_settings('saveRAW', self.camera_profile['saveRAW'], True)

            self.reconfigure_video_pipeline()
            self.apply_profile_controls()
            self.sync_live_controls()  # Ensure UI updates with the latest settings

            # Update camera-last-config.json
            try:
                if os.path.exists(self.last_config_path):
                    with open(self.last_config_path, "r") as f:
                        last_config = json.load(f)
                else:
                    last_config = {"cameras": []}
                # Find the matching camera entry
                updated = False
                for camera in last_config["cameras"]:
                    if camera["Num"] == self.camera_num:
                        camera["Has_Config"] = True
                        camera["Config_Location"] = profile_filename
                        updated = True
                        break
                if not updated:
                    print(f"Camera {self.camera_num} not found in camera-last-config.json.")
                with open(self.last_config_path, "w") as f:
                    json.dump(last_config, f, indent=4)
                print(f"Loaded profile '{profile_filename}' and updated camera-last-config.json.")
            except Exception as e:
                print(f"Error updating camera-last-config.json: {e}")
            
            print(f"Sucessfully loaded profile {profile_path}")
            return True
        except Exception as e:
            print(f"Error loading camera profile '{profile_filename}': {e}")
            return False

    def generate_camera_profile(self):
        file_name = os.path.join(camera_profile_folder, 'camera-module-info.json')
        # If there is no existing config, or the file doesn't exist, create a default profile
        if not self.camera_info.get("Has_Config", False) or not os.path.exists(file_name):
            self.camera_profile = {
                "hflip": 0,
                "vflip": 0,
                "sensor_mode": 0,
                "model": self.camera_info.get("Model", "Unknown"),
                "resolutions": {
                    "StillCaptureResolution": 0,
                    "recording_resolution": 0,
                    "streaming_resolution": 0
                },
                "saveRAW": False,
                "controls": {}
            }
        else:
            # Load existing profile from file
            with open(file_name, 'r') as file:
                self.camera_profile = json.load(file)
        return self.camera_profile
    
    def initialize_controls_template(self, picamera2_controls, camera_control_db_path):
        if os.path.isfile(camera_control_db_path):
            try:
                with open(camera_control_db_path, "r") as f:
                    cam_ctrl_json = json.load(f)
            except Exception as e:
                print(f"Error: Failed to extract json data from '{camera_control_db_path}': {e}")
                return {}
            if "sections" not in cam_ctrl_json:
                print("Error: 'sections' key not found in cam_ctrl_json!")
                return {cam_ctrl_json}  # Return unchanged if it's not structured as expected
        else:
            print(f"Error: file {camera_control_db_path} doesn't exist")
            return {}
        # Initialize empty controls in camera_profile
        self.camera_profile["controls"] = {}
        for section in cam_ctrl_json["sections"]:
            if "settings" not in section:
                print(f"Warning: Missing 'settings' key in section: {section.get('title', 'Unknown')}")
                continue        
            section_enabled = False  # Track if any setting is enabled
            for setting in section["settings"]:
                if not isinstance(setting, dict):
                    print(f"Warning: Unexpected setting format: {setting}")
                    continue  # Skip if it's not a dictionary
                setting_id = setting.get("id")  # Use `.get()` to avoid crashes
                source = setting.get("source", None)  # Check if source exists
                original_enabled = setting.get("enabled", False)  # Preserve original enabled state
                
                if source == "controls":
                    if setting_id in picamera2_controls:
                        min_val, max_val, default_val = picamera2_controls[setting_id]
                        print(f"Updating {setting_id}: Min={min_val}, Max={max_val}, Default={default_val}")  # Debugging
                        setting["min"] = min_val
                        setting["max"] = max_val
                        if default_val is not None:
                            setting["default"] = default_val
                        else:
                            default_val = False if isinstance(min_val, bool) else min_val                        
                        if setting["enabled"]:
                            self.camera_profile["controls"][setting_id] = default_val
                        setting["enabled"] = original_enabled  
                        if original_enabled:
                            section_enabled = True                 
                    else:
                        print(f"Disabling {setting_id}: Not found in picamera2_controls")  # Debugging
                        setting["enabled"] = False  # Disable setting         
                elif source == "generatedresolutions":
                    resolution_options = [
                        {"value": i, "label": f"{w} x {h}", "enabled": True}
                        for i, (w, h) in enumerate(self.camera_resolutions)
                    ]
                    # Use the dynamically generated resolutions
                    setting["options"] = resolution_options
                    section_enabled = True
                    print(f"Updated {setting_id} with generated resolutions")
                elif source == "video_resolutions":
                    resolution_options = [
                        {"value": i, "label": f"{w} x {h}", "enabled": True}
                        for i, (w, h) in enumerate(self.video_resolutions)
                    ]
                    setting["options"] = resolution_options
                    section_enabled = True
                else:
                    print(f"Skipping {setting_id}: No source specified, keeping existing values.")
                    section_enabled = True  
            
                if "childsettings" in setting:
                    for child in setting["childsettings"]:
                        child_id = child.get("id")
                        child_source = child.get("source", None)
                        if child_source == "controls" and child_id in picamera2_controls:
                            min_val, max_val, default_val = picamera2_controls[child_id]
                            print(f"Updating Child Setting {child_id}: Min={min_val}, Max={max_val}, Default={default_val}")  # Debugging
                            child["min"] = min_val
                            child["max"] = max_val
                            self.camera_profile["controls"][child_id] = default_val if default_val is not None else min_val
                            if default_val is not None:
                                child["default"] = default_val  
                            child["enabled"] = child.get("enabled", False)
                            if child["enabled"]:
                                section_enabled = True  
                        else:
                            print(f"Skipping or Disabling Child Setting {child_id}: Not found or no source specified")
            section["enabled"] = section_enabled
        print(f"Initialized camera_profile controls: {self.camera_profile}")
        return cam_ctrl_json

    def update_settings(self, setting_id, setting_value, init=False):
        # Handle hflip and vflip separately
        if setting_id in ["hflip", "vflip"]:
            try:
                self.camera_profile[setting_id] = bool(int(setting_value))
                if not init:
                    self.reconfigure_video_pipeline()

                print(f"Applied transform: {setting_id} -> {setting_value} (Camera restarted)")
            except ValueError as e:
                print(f"⚠️ Error: {e}")
        elif setting_id == "StillCaptureResolution":
            self.camera_profile['resolutions'][setting_id] = int(setting_value)
            print(f"Applied StillCaptureResolution index {setting_value} -> {self.camera_resolutions[int(setting_value)]}")

        # -----------------------------
        # Video Resolutions: Recording / Streaming
        # -----------------------------

        elif setting_id == "recording_resolution":
            self.set_recording_resolution(int(setting_value))
            self.camera_profile['resolutions'][setting_id] = int(setting_value)
            print(f"Applied recording_resolution index {setting_value} -> {self.video_resolutions[int(setting_value)]}")
            if not init:
                self.reconfigure_video_pipeline()

        elif setting_id == "streaming_resolution":
            self.set_streaming_resolution(int(setting_value))
            self.camera_profile['resolutions'][setting_id] = int(setting_value)
            print(f"Applied streaming_resolution index {setting_value} -> {self.video_resolutions[int(setting_value)]}")
            if not init:
                self.reconfigure_video_pipeline()

        elif setting_id == "saveRAW":
            try:
                self.camera_profile[setting_id] = setting_value
                print(f"Applied transform: {setting_id} -> {setting_value}")
            except ValueError as e:
                print(f"[ERROR]: {e}")
        else:
            # Convert setting_value to correct type
            if "." in str(setting_value):
                setting_value = float(setting_value)
            else:
                setting_value = int(setting_value)
            # Apply the setting
            self.picam2.set_controls({setting_id: setting_value})
            # Store in camera_profile["controls"]
            self.camera_profile.setdefault("controls", {})[setting_id] = setting_value
        # Update live settings
        updated = False
        for section in self.live_controls.get("sections", []):
            for setting in section.get("settings", []):
                if setting["id"] == setting_id:
                    setting["value"] = setting_value  # Update main setting
                    updated = True
                    break
                # Check child settings
                for child in setting.get("childsettings", []):
                    if child["id"] == setting_id:
                        child["value"] = setting_value  # Update child setting
                        updated = True
                        break
            if updated:
                break  # Exit loop once found
        if not updated:
            print(f"[WARNING]: Setting {setting_id} not found in live_controls!")
        return setting_value  # Returning for confirmation

    def sync_live_controls(self):
        """Updates self.live_controls to match self.camera_profile without resetting defaults."""
        for section in self.live_controls.get("sections", []):
            for setting in section.get("settings", []):
                setting_id = setting["id"]
                if setting_id in self.camera_profile["controls"]:
                    setting["value"] = self.camera_profile["controls"][setting_id]
                # Sync child settings
                for child in setting.get("childsettings", []):
                    child_id = child["id"]
                    if child_id in self.camera_profile["controls"]:
                        child["value"] = self.camera_profile["controls"][child_id]
        print("Live controls updated to match camera profile.")

    def apply_profile_controls(self):
        if "controls" in self.camera_profile:
            try:
                for setting_id, setting_value in self.camera_profile["controls"].items():
                    self.picam2.set_controls({setting_id: setting_value})
                    self.update_settings(setting_id, setting_value)  # Use the loop variables
                    print(f"Applied Control: {setting_id} -> {setting_value}")
                print("All profile controls applied successfully")
            except Exception as e:
                print(f"[ERROR] applying profile controls: {e}")

    def generate_video_resolutions(self):
        resolutions = set()

        for mode in self.sensor_modes:
            sw, sh = mode["size"]

            # resolution candidates - may be changed
            for w, h in [
                (1920, 1080),
                (1536, 864),
                (1280, 720),
                (1152, 648),
                (768, 432),
            ]:
                if w <= ENCODER_MAX[0] and h <= ENCODER_MAX[1]:
                    if w <= sw and h <= sh:
                        resolutions.add((w, h))

        return sorted(resolutions, reverse=True)

    def find_best_sensor_mode(self, video_resolution):
        tw, th = video_resolution

        candidates = [
            m for m in self.sensor_modes
            if m["size"][0] >= tw and m["size"][1] >= th
        ]

        if not candidates:
            raise ValueError("No suitable sensor mode found")

        # priorice smallest resolution
        return min(candidates, key=lambda m: m["size"][0] * m["size"][1])

    def reconfigure_video_pipeline(self):
        if self.active_capture_still or self.active_recording:
            print("Failed to reconfigure video pipeline since there is an active still capture or recording")
            return False
        rec_index = self.camera_profile["resolutions"]["recording_resolution"]
        stream_index = self.camera_profile["resolutions"]["streaming_resolution"]

        rec = self.video_resolutions[rec_index]
        stream = self.video_resolutions[stream_index]

        # define larger resolution
        if rec[0] * rec[1] >= stream[0] * stream[1]:
            main_size = rec
            lores_size = stream
            self.main_stream = "recording"
            self.lores_stream = "streaming"
        else:
            main_size = stream
            lores_size = rec
            self.main_stream = "streaming"
            self.lores_stream = "recording"

        # define apropriate sensor mode
        mode = self.find_best_sensor_mode(main_size)

        was_streaming = self.active_stream
        if was_streaming:
            self.stop_streaming()

        with self.lock:
            self.picam2.stop()

            transform = Transform()
            transform.hflip = bool(self.camera_profile.get("hflip", False))
            transform.vflip = bool(self.camera_profile.get("vflip", False))

            self.video_config = self.picam2.create_video_configuration(
                main={"size": main_size},
                lores={"size": lores_size},
                transform=transform,
                sensor={
                    "output_size": mode["size"],
                    "bit_depth": mode["bit_depth"]
                }
            )

            self.picam2.configure(self.video_config)
            self.picam2.start()

        if was_streaming:
            self.start_streaming()

        # update sensor mode in camera profile
        self.camera_profile["sensor_mode"] = self.sensor_modes.index(mode)

        print("✅ Video pipeline reconfigured")
        print(f"   main:  {main_size} ({self.main_stream})")
        print(f"   lores: {lores_size} ({self.lores_stream})")

        return True

    def get_recording_stream(self):
        return "main" if self.main_stream == "recording" else "lores"

    def get_streaming_stream(self):
        return "main" if self.main_stream == "streaming" else "lores"


    def get_recording_resolution(self):
        return self.video_resolutions[
            self.camera_profile["resolutions"]["recording_resolution"]
        ]

    def get_streaming_resolution(self):
        return self.video_resolutions[
            self.camera_profile["resolutions"]["streaming_resolution"]
        ]

    def set_recording_resolution(self, resolution_index: int):
        self.camera_profile["resolutions"]["recording_resolution"] = int(resolution_index)
        self.reconfigure_video_pipeline()

    def set_streaming_resolution(self, resolution_index: int):
        self.camera_profile["resolutions"]["streaming_resolution"] = int(resolution_index)
        self.reconfigure_video_pipeline()

    def create_profile(self, filename):
        """Save the current camera profile and update camera-last-config.json."""
        try:
            print(self.camera_profile)
            # Ensure .json is not already in the filename
            if filename.lower().endswith(".json"):
                filename = filename[:-5]
            profile_path = os.path.join(camera_profile_folder, f"{filename}.json")
            # Save the profile
            with open(profile_path, "w") as f:
                json.dump(self.camera_profile, f, indent=4)
            # ✅ Update camera-last-config.json
            try:
                if os.path.exists(self.last_config_path):
                    with open(self.last_config_path, "r") as f:
                        last_config = json.load(f)
                else:
                    last_config = {"cameras": []}  # Create an empty structure if missing
                # Find the camera entry matching the current camera number
                updated = False
                for camera in last_config["cameras"]:
                    if camera["Num"] == self.camera_num:
                        camera["Has_Config"] = True
                        camera["Config_Location"] = f"{filename}.json"  # Set the new config file
                        updated = True
                        break
                if not updated:
                    print(f"Warning: Camera {self.camera_num} not found in camera-last-config.json.")
                # Save the updated configuration back
                with open(self.last_config_path, "w") as f:
                    json.dump(last_config, f, indent=4)
                print(f"Updated camera-last-config.json for camera {self.camera_num} after saving profile.")
            except Exception as e:
                print(f"Error updating camera-last-config.json: {e}")
            return True
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False

    def reset_to_default(self):
        # Resets camera settings to default and applies them.
        self.camera_profile = {
            "hflip": 0,
            "vflip": 0,
            "model": self.camera_info.get("Model", "Unknown"),
            "resolutions": {
                "StillCaptureResolution": 0,
                "recording_resolution": 0,
                "streaming_resolution": 0
            },
            "saveRAW": False,
            "controls": {
                "AfMode": 0,
                "LensPosition": 1.0,
                "AfRange": 0,
                "AfSpeed": 0,
                "ExposureTime": 33000,
                "AnalogueGain": 1.1228070259094238,
                "AeEnable": 1,
                "ExposureValue": 0.0,
                "AeConstraintMode": 0,
                "AeExposureMode": 0,
                "AeMeteringMode": 0,
                "AeFlickerMode": 0,
                "AeFlickerPeriod": 100,
                "AwbEnable": 0,
                "AwbMode": 0,
                "Brightness": 0,
                "Contrast": 1.0,
                "Saturation": 1.0,
                "Sharpness": 1.0,
                "ColourTemperature": 4000
            }
        }

        # Apply video resolutions and orientation
        self.reconfigure_video_pipeline()
        # Reinitialize UI settings
        self.update_settings("saveRAW", self.camera_profile["saveRAW"])
        # Apply camera controls
        self.apply_profile_controls()
        print("Camera profile reset to default and settings applied.")
        return True

    #-----
    # Camera Information Functions
    #-----

    def capture_metadata(self):
        self.metadata = self.picam2.capture_metadata()
        #print(f"Metadata: {self.metadata}")
        print(self.picam2.sensor_resolution)
        return self.metadata

    def get_camera_module_spec(self):
        """Return camera module details for this camera."""
        camera_module = next(
            (cam for cam in self.camera_module_info.get("camera_modules", [])
            if cam["sensor_model"] == self.camera_info["Model"]),
            None
        )
        return camera_module

    def get_sensor_mode(self):
        current_config = self.picam2.camera_configuration()
        active_mode = current_config.get('sensor', {})  # Get the currently active sensor settings
        active_mode_index = None  # Default to None if no match is found
        # Find the matching sensor mode index
        for index, mode in enumerate(self.sensor_modes):
            if mode['size'] == active_mode.get('output_size') and mode['bit_depth'] == active_mode.get('bit_depth'):
                active_mode_index = index
                break
        print(f"Active Sensor Mode: {active_mode_index}")
        return active_mode_index

    def generate_camera_resolutions(self):
        """
        Precompute a list of resolutions based on the available sensor modes.
        This list is shared between still capture and live feed resolution settings.
        """
        if not self.sensor_modes:
            print("[WARNING]: No sensor modes available!")
            return []

        # Extract sensor mode resolutions
        resolutions = sorted(set(mode['size'] for mode in self.sensor_modes if 'size' in mode), reverse=True)

        if not resolutions:
            print("[WARNING]: No valid resolutions found in sensor modes!")
            return []

        max_resolution = resolutions[0]  # Highest resolution
        aspect_ratio = max_resolution[0] / max_resolution[1]

        # Generate midpoint resolutions
        extra_resolutions = []
        for i in range(len(resolutions) - 1):
            w1, h1 = resolutions[i]
            w2, h2 = resolutions[i + 1]
            midpoint = ((w1 + w2) // 2, (h1 + h2) // 2)
            extra_resolutions.append(midpoint)

        # Add two extra smaller resolutions at the end
        last_w, last_h = resolutions[-1]
        half_res = (last_w // 2, last_h // 2)
        inbetween_res = ((last_w + half_res[0]) // 2, (last_h + half_res[1]) // 2)

        resolutions.extend(extra_resolutions)
        resolutions.append(inbetween_res)
        resolutions.append(half_res)

        # Store in camera object for later use
        self.available_resolutions = sorted(set(resolutions), reverse=True)

        return self.available_resolutions

    #-----
    # Camera Streaming Functions
    #-----

    def start_streaming(self):
        if self.active_stream:
            print("[INFO] Skip starting stream, since there is already an active stream")
            return

        stream_name = self.get_streaming_stream()

        with self.lock:
            self.picam2.start_recording(
                self.encoder_stream,
                output=self.output_stream,
                name=stream_name
            )

            self.active_stream = True
            print(f"[INFO] Starting streaming on '{stream_name}' stream")

    def stop_streaming(self):
        if not self.active_stream:  # Skip if there is no active stream
            print("[INFO] Skip stopping stream, since there is no active stream")
            return
        else:
            with self.lock:
                self.picam2.stop_recording()
                self.active_stream = False
                print("[INFO] Streaming stopped")

    #-----
    # Camera Recording Functions
    #-----

    def start_recording(self, filename_recording):
        if self.active_recording:
            print("[INFO] Skip starting recording, since there is already an active recording")
            return False, None

        stream_name = self.get_recording_stream()
        filepath = os.path.join(self.upload_folder, filename_recording)
        output_recording = FfmpegOutput(filepath, audio=self.audio)

        try:
            with self.lock:
                self.picam2.start_recording(
                    self.encoder_recording,
                    output_recording,
                    name=stream_name
                )

                self.output_recording = output_recording
                self.active_recording = True
                self.filename_recording = filename_recording
                print(f"[INFO] Recording {filename_recording} started")

                return True, filename_recording
        except Exception as e:
            print(f"[ERROR] Failed to start recording {filename_recording}: {e}")
            return False            

    def stop_recording(self):
        if not self.active_recording:
            print("[INFO] Skip stopping recording, since there is no active recording")
            return False
        with self.lock:
            self.picam2.stop_recording()
            self.output_recording = None
            self.active_recording = False
            self.active_stream = False

            print(f"[INFO] Recording {self.filename_recording} stopped")

        self.start_streaming()

        return True

    #-----
    # Camera Capture Functions
    #-----

    def capture_still(self, filepath, raw=False):
        with self.lock:
            if self.active_capture_still:
                print(f"Skip to capture still image {filepath}, since there is another active process capturing a still image on this camera")
                return None
            
            else:
                self.active_capture_still = True

        still_resoultion_index = self.camera_profile['resolutions']['StillCaptureResolution']
        still_resolution = self.camera_resolutions[still_resoultion_index]

        recording_resolution_index = self.camera_profile["resolutions"]["recording_resolution"]
        recording_resolution = self.video_resolutions[recording_resolution_index]

        # set sensor mode based on higher resolution (streaming resolution or still image resolution) to guarantee, that the FOV of the still image is not smaller than the stream shown in webui (FOV still image >= FOV stream)
        if still_resolution[0] * still_resolution[1] >= recording_resolution[0] * recording_resolution[1]:
            mode = self.find_best_sensor_mode(still_resolution)
        else:
            mode = self.find_best_sensor_mode(recording_resolution)

        still_config = self.picam2.create_still_configuration(buffer_count=1, main={"size": still_resolution}, sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]}, controls={"FrameDurationLimits": (100, 10000000000)})

        self.stop_recording()
        self.stop_streaming()

        with self.lock:
            self.picam2.stop()
            self.picam2.configure(still_config)
            self.camera_profile["sensor_mode"] = self.sensor_modes.index(mode)
            self.picam2.start()
            # camera.picam2.capture_file(filepath)

            if raw:
                print(f"DEBUG: Start capturing raw image {filepath}")
                # This will be the new way to save images at max quality just need to make the save as DNG setting available
                buffers, metadata = self.picam2.switch_mode_and_capture_buffers(still_config, ["main", "raw"])
                self.picam2.helpers.save(self.picam2.helpers.make_image(buffers[0], still_config["main"]), metadata, filepath)
                self.picam2.helpers.save_dng(buffers[1], metadata, still_config["raw"], filepath)
            else:
                print(f"DEBUG: Start capturing non-raw image {filepath}")
                # This will be the new way to save images at max quality just need to make the save as DNG setting available
                buffers, metadata = self.picam2.switch_mode_and_capture_buffers(still_config, ["main"])
                self.picam2.helpers.save(self.picam2.helpers.make_image(buffers[0], still_config["main"]), metadata, filepath)

            logging.info(f"Successuffly captured image {filepath}")
            self.active_capture_still = False
            return filepath

    def capture_still_from_feed(self, filepath):
        try:
            request = self.picam2.capture_request()
            request.save("main", f'{filepath}')
            print(f"Image captured successfully. Path: {filepath}")
            return f'{filepath}.jpg'
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None