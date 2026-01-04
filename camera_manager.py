import os
import json
import threading
import logging
from typing import Dict, List, Optional
from picamera2 import Picamera2
from camera_object import CameraObject

# Flask-SocketIO Imports
from flask_socketio import SocketIO, emit, join_room, leave_room

logger = logging.getLogger(__name__)

####################
# CameraManager Class
####################

class CameraManager:
    def __init__(
        self,
        camera_module_info_path: str,
        last_config_path: str,
        media_upload_folder: str,
        camera_controls_db_path: str,
        camera_profile_folder: str,
        socketio: SocketIO,
    ):
        """
        :param camera_module_info_path: Path to camera-module-info.json
        :param last_config_path: Path to camera-last-config.json
        :param media_upload_folder: Path to the folder where photos and videos are stored (media gallery)
        :camera_controls_db_path: Path to file storing camera controls parameter, controllable via webui
        :camera_profile_folder: Path to folder storing files of saved camera profiles (.json)
        :socketio: Flask-SocketIO object used to snyc camera settings
        """

        try:
            with open(camera_module_info_path, "r") as f:
                self.camera_module_info = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load camera module info from '%s': %s",
                camera_module_info_path,
                exc,
            )
            self.camera_module_info = {}

        self.last_config_path = last_config_path
        self.media_upload_folder = media_upload_folder
        self.camera_controls_db_path = camera_controls_db_path
        self.camera_profile_folder = camera_profile_folder
        self.socketio = socketio

        self.connected_cameras: List[dict] = []
        self.cameras: Dict[int, CameraObject] = {}
        self.lock = threading.Lock()

    def _load_last_config(self):
        """Load the last configuration file or create an empty template."""
        if os.path.exists(self.last_config_path):
            with open(self.last_config_path, "r") as f:
                self.last_config = json.load(f)
            logger.debug("Loaded last camera configuration from disk.")
        else:
            self.last_config = {"cameras": []}
            logger.info("No previous camera configuration found. Using empty template.")

    def _detect_connected_cameras(self) -> List[dict]:
        """
        Detect currently connected cameras using Picamera2 and
        determine whether they are Raspberry Pi cameras.
        """
        currently_connected = []

        for connected_camera in Picamera2.global_camera_info():
            matching_module = next(
                (
                    module
                    for module in self.camera_module_info.get("camera_modules", [])
                    if module["sensor_model"] == connected_camera["Model"]
                ),
                None,
            )

            is_pi_cam = bool(matching_module and matching_module.get("is_pi_cam", False))

            if is_pi_cam:
                logger.info(
                    "Detected Raspberry Pi Camera: model=%s",
                    connected_camera["Model"],
                )
            else:
                logger.info(
                    "Detected non-Pi camera or unknown model: model=%s",
                    connected_camera["Model"],
                )

            camera_info = {
                "Num": connected_camera["Num"],
                "Model": connected_camera["Model"],
                "Is_Pi_Cam": is_pi_cam,
                "Has_Config": False,
                "Config_Location": f"default_{connected_camera['Model']}.json",
            }

            currently_connected.append(camera_info)

        self.connected_cameras = currently_connected
        return currently_connected

    def _sync_last_config(self) -> List[dict]:
        """
        Compare currently connected cameras with the last configuration
        and update it if necessary.
        """
        existing_lookup = {
            cam["Num"]: cam for cam in self.last_config.get("cameras", [])
        }

        updated_cameras = []

        for cam in self.connected_cameras:
            cam_num = cam["Num"]

            if cam_num in existing_lookup:
                config_cam = existing_lookup[cam_num]

                if (
                    config_cam["Model"] != cam["Model"]
                    or config_cam.get("Is_Pi_Cam") != cam.get("Is_Pi_Cam")
                ):
                    logger.info(
                        "Camera %s changed (model or Pi Camera flag). Updating config.",
                        cam_num,
                    )
                    updated_cameras.append(cam)
                else:
                    updated_cameras.append(config_cam)
            else:
                logger.info("New camera detected and added to config: %s", cam)
                updated_cameras.append(cam)

        self.last_config = {"cameras": updated_cameras}
        with open(self.last_config_path, "w") as f:
            json.dump(self.last_config, f, indent=4)

        self.connected_cameras = updated_cameras
        return updated_cameras

    def _make_state_callback(self, camera):
        def callback():
            state = camera.get_state()
            room = f"camera_{camera.camera_num}"
            print(f"DEBUG: _make_state_callback, state: {state}, room: {room}")
            self.socketio.emit(
                "camera_state",
                {"camera_num": camera.camera_num, "state": state},
                room=room,
            )
        return callback

    def join_camera_room(self, sid, camera_num):
        """ SocketIO-Client joins camera room """
        room = f"camera_{camera_num}"
        join_room(room, sid=sid)

    def leave_camera_room(self, sid, camera_num):
        """ SocketIO-Client leaves camera room """
        room = f"camera_{camera_num}"
        leave_room(room, sid=sid)

    def init_cameras(self):
        """Create CameraObject instances for all connected cameras."""
        self._load_last_config()
        self._detect_connected_cameras()
        self._sync_last_config()

        for cam_info in self.connected_cameras:
            try:
                camera_obj = CameraObject(
                    cam_info,
                    self.camera_module_info,
                    self.media_upload_folder,
                    self.last_config_path,
                    self.camera_controls_db_path,
                    self.camera_profile_folder,
                )
                camera_obj._on_state_changed = self._make_state_callback(camera_obj)
                self.cameras[cam_info["Num"]] = camera_obj
            except Exception as e:
                logger.error("Failed to initialize camera %s: %s", cam_info["Num"], e)

        for key, camera in self.cameras.items():
            logger.info("Initialized camera %s: %s", key, camera.camera_info)

    def get_camera(self, cam_num: int) -> Optional[CameraObject]:
        """
        Return the CameraObject instance for the given camera number.
        Returns None if the request is invalid or the camera does not exist.
        """
        if not isinstance(cam_num, int):
            logger.warning("Invalid camera number requested: %r", cam_num)
            return None

        return self.cameras.get(cam_num)

    def list_cameras(self) -> Optional[List[dict]]:
        """
        Return a list of all connected cameras as dictionaries.
        Returns None if no cameras are present or the internal state is invalid.
        """
        if not isinstance(self.connected_cameras, list):
            logger.error("Invalid internal state: connected_cameras is not a list.")
            return None

        if not self.connected_cameras:
            logger.info("No connected cameras detected.")
            return None

        return self.connected_cameras