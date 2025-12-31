
import os
import json
import threading
from typing import Dict, List
from camera_object import CameraObject  # Annahme: CameraObject ist in camera_object.py

####################
# CameraManager Class
####################

class CameraManager:
    def __init__(self, global_cameras: List[dict], camera_module_info: dict, config_path: str, upload_folder: str):
        """
        :param global_cameras: Liste aller aktuell erkannten Kameras [{'Num': int, 'Model': str}, ...]
        :param camera_module_info: JSON-Daten aus camera-module-info.json
        :param config_path: Pfad zur camera-last-config.json
        """
        self.global_cameras = global_cameras
        self.camera_module_info = camera_module_info
        self.config_path = config_path
        self.upload_folder = upload_folder

        self.currently_connected_cameras: List[dict] = []
        self.cameras: Dict[int, CameraObject] = {}
        self.lock = threading.Lock()

        self._load_last_config()

    def _load_last_config(self):
        """Lade das letzte Config-File oder erstelle leeres Template."""
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self.last_config = json.load(f)
        else:
            self.last_config = {"cameras": []}

    def detect_connected_cameras(self) -> List[dict]:
        """Erstelle die Liste der aktuell verbundenen Kameras mit Pi-Cam Kennzeichnung."""
        currently_connected = []
        for connected_camera in self.global_cameras:
            matching_module = next(
                (module for module in self.camera_module_info.get("camera_modules", [])
                 if module["sensor_model"] == connected_camera["Model"]),
                None
            )
            is_pi_cam = bool(matching_module and matching_module.get("is_pi_cam", False))
            if is_pi_cam:
                print(f"Connected camera model '{connected_camera['Model']}' is a Pi Camera.")
            else:
                print(f"Connected camera model '{connected_camera['Model']}' is NOT a Pi Camera or not listed.")

            camera_info = {
                "Num": connected_camera["Num"],
                "Model": connected_camera["Model"],
                "Is_Pi_Cam": is_pi_cam,
                "Has_Config": False,
                "Config_Location": f"default_{connected_camera['Model']}.json"
            }
            currently_connected.append(camera_info)

        self.currently_connected_cameras = currently_connected
        return currently_connected

    def sync_last_config(self) -> List[dict]:
        """Vergleiche aktuelle Kameras mit der letzten Konfiguration und aktualisiere, falls nötig."""
        existing_lookup = {cam["Num"]: cam for cam in self.last_config.get("cameras", [])}
        updated_cameras = []

        for new_cam in self.currently_connected_cameras:
            cam_num = new_cam["Num"]
            if cam_num in existing_lookup:
                old_cam = existing_lookup[cam_num]
                # Update wenn sich Model oder Pi-Cam Status geändert hat
                if old_cam["Model"] != new_cam["Model"] or old_cam.get("Is_Pi_Cam") != new_cam.get("Is_Pi_Cam"):
                    print(f"Updating camera {new_cam['Model']}: Model or Pi-Cam status changed.")
                    updated_cameras.append(new_cam)
                else:
                    updated_cameras.append(old_cam)
            else:
                # Neue Kamera
                print(f"New camera added to config: {new_cam}")
                updated_cameras.append(new_cam)

        # Save updated config
        self.last_config = {"cameras": updated_cameras}
        with open(self.config_path, "w") as f:
            json.dump(self.last_config, f, indent=4)

        self.currently_connected_cameras = updated_cameras
        return updated_cameras

    # init_cameras-Methode im CameraManager
    def init_cameras(self):
        """Erstelle CameraObject Instanzen für alle verbundenen Kameras."""
        for cam_info in self.currently_connected_cameras:
            # Übergabe der Kamera-Info und des camera_module_info
            camera_obj = CameraObject(cam_info, self.camera_module_info, self.upload_folder)
            self.cameras[cam_info["Num"]] = camera_obj

        # Debug-Ausgabe
        for key, camera in self.cameras.items():
            print(f"Key: {key}, Camera: {camera.camera_info}")

    def get_camera(self, cam_num: int) -> CameraObject:
        """Hole die CameraObject Instanz für eine bestimmte Kamera."""
        return self.cameras.get(cam_num)

    def list_cameras(self) -> List[dict]:
        """Liste aller verbundenen Kameras als Dictionaries."""
        return self.currently_connected_cameras