
import os
import json
import threading
from typing import Dict, List, Optional
from picamera2 import Picamera2
from camera_object import CameraObject  # Annahme: CameraObject ist in camera_object.py

####################
# CameraManager Class
####################

class CameraManager:
    def __init__(self, camera_module_info_path: str, last_config_path: str, media_upload_folder: str, camera_control_db_path: str):
        """
        :param camera_module_info_path: Pfad zur Datei camera-module-info.json
        :param last_config_path: Pfad zur camera-last-config.json
        :param media_upload_folder: Pfad zum Ordner in welchem die Fotos und Videos-Aufnahmen gesichert werden (media gallery)
        """

        try:
            with open(camera_module_info_path, 'r') as f:
                self.camera_module_info = json.load(f)
        except:
            self.camera_module_info = {}
        
        self.last_config_path = last_config_path
        self.media_upload_folder = media_upload_folder
        self.camera_control_db_path = camera_control_db_path

        self.connected_cameras: List[dict] = []
        self.cameras: Dict[int, CameraObject] = {}
        self.lock = threading.Lock()

    def init_cameras(self):
        """Erstelle CameraObject Instanzen für alle verbundenen Kameras."""
        self._load_last_config()
        self._detect_connected_cameras()
        self._sync_last_config()

        for cam_info in self.connected_cameras:
            # Übergabe der Kamera-Info und des camera_module_info
            camera_obj = CameraObject(cam_info, self.camera_module_info, self.media_upload_folder, self.last_config_path, self.camera_control_db_path)
            self.cameras[cam_info["Num"]] = camera_obj

        # Debug-Ausgabe
        for key, camera in self.cameras.items():
            print(f"Key: {key}, Camera: {camera.camera_info}")

    def _load_last_config(self):
        """Lade das letzte Config-File oder erstelle leeres Template."""
        if os.path.exists(self.last_config_path):
            with open(self.last_config_path, "r") as f:
                self.last_config = json.load(f)
        else:
            self.last_config = {"cameras": []}

    def _detect_connected_cameras(self) -> List[dict]:
        """Erstelle die Liste der aktuell verbundenen Kameras (erkannt durch Picamera2) mit Pi-Cam Kennzeichnung."""
        currently_connected = []
        for connected_camera in Picamera2.global_camera_info():
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

        self.connected_cameras = currently_connected
        return currently_connected

    def _sync_last_config(self) -> List[dict]:
        """Vergleiche aktuelle Kameras mit der letzten Konfiguration und aktualisiere, falls nötig."""
        existing_lookup = {cam["Num"]: cam for cam in self.last_config.get("cameras", [])}
        updated_cameras = []

        for cam in self.connected_cameras:
            cam_num = cam["Num"]
            if cam_num in existing_lookup:
                config_cam = existing_lookup[cam_num]
                # Update wenn sich Model oder Pi-Cam Status geändert hat
                if config_cam["Model"] != cam["Model"] or config_cam.get("Is_Pi_Cam") != cam.get("Is_Pi_Cam"):
                    print(f"Updating camera {cam['Model']}: Model or Pi-Cam status changed.")
                    updated_cameras.append(cam)
                else:
                    updated_cameras.append(config_cam)
            else:
                # Neue Kamera
                print(f"New camera added to config: {cam}")
                updated_cameras.append(cam)

        # Save updated config
        self.last_config = {"cameras": updated_cameras}
        with open(self.last_config_path, "w") as f:
            json.dump(self.last_config, f, indent=4)

        self.connected_cameras = updated_cameras
        return updated_cameras

    def get_camera(self, cam_num: int) -> Optional[CameraObject]:
        """Hole die CameraObject-Instanz für eine bestimmte Kamera.
        Gibt None zurück, falls die Abfrage ungültig ist oder die Kamera nicht existiert.
        """
        if not isinstance(cam_num, int):
            return None

        camera = self.cameras.get(cam_num)
        return camera if camera is not None else None

    def list_cameras(self) -> Optional[List[dict]]:
        """Liste aller verbundenen Kameras als Dictionaries.
        Gibt None zurück, falls keine Kameras vorhanden sind oder der Zustand ungültig ist.
        """
        if not isinstance(self.connected_cameras, list):
            return None

        if not self.connected_cameras:
            return None

        return self.connected_cameras