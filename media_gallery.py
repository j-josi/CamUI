import json
import os
import logging
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, ImageOps, ImageEnhance

####################
# MediaGallery Class
####################

logger = logging.getLogger(__name__)

class MediaGallery:
    def __init__(self, upload_folder: str):
        self.upload_folder: str = upload_folder
        self.image_exts: Tuple[str, ...] = ('.jpg', '.jpeg')
        self.video_exts: Tuple[str, ...] = ('.mp4',)

    def get_image_resolution(self, path: str) -> Tuple[int, int]:
        with Image.open(path) as img:
            width, height = img.size
        return width, height

    def get_video_resolution(self, path: str) -> Tuple[Optional[int], Optional[int]]:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "json",
                    path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            data = json.loads(result.stdout)
            stream = data.get("streams", [{}])[0]
            return stream.get("width"), stream.get("height")
        except Exception as e:
            logger.warning(f"Could not read video resolution for {path}: {e}")
            return None, None

    def get_media_files(self, type: str = "all", excluded_files: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            excluded_files = excluded_files or []

            files = os.listdir(self.upload_folder)
            media: List[Dict[str, Any]] = []

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if f in excluded_files:
                    continue
                if type == "all" and ext not in self.image_exts + self.video_exts:
                    continue
                elif type == "image" and ext not in self.image_exts:
                    continue
                elif type == "video" and ext not in self.video_exts:
                    continue
                elif type not in ["all", "image", "video"]:
                    continue

                path = os.path.join(self.upload_folder, f)
                item: Dict[str, Any] = {
                    "filename": f,
                    "type": "video" if ext in self.video_exts else "image",
                    "width": None,
                    "height": None,
                    "has_dng": False,
                    "dng_file": None
                }

                if item["type"] == "image":
                    item["width"], item["height"] = self.get_image_resolution(path)
                    dng = os.path.splitext(f)[0] + ".dng"
                    item["has_dng"] = os.path.exists(os.path.join(self.upload_folder, dng))
                    item["dng_file"] = dng
                elif item["type"] == "video":
                    item["width"], item["height"] = self.get_video_resolution(path)

                media.append(item)

            media.sort(key=lambda x: x["filename"], reverse=True)
            return media

        except Exception as e:
            logger.error(f"Media loading error: {e}")
            return []

    def get_media_slice(self, offset: int = 0, limit: int = 20, type: str = "all", excluded_files: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return a slice of media for infinite scroll."""
        all_media = self.get_media_files(type=type, excluded_files=excluded_files)
        return all_media[offset:offset + limit]

    def find_last_image_taken(self) -> Optional[str]:
        """Find the most recent image taken."""
        all_images = self.get_media_files(type="image")
        if all_images:
            first_image = all_images[0]
            logger.debug(f"Last image found: {first_image['filename']}")
            return first_image['filename']
        else:
            logger.debug("No image files found.")
            return None

    def apply_filter(
        self,
        filepath: str,
        rotation: Optional[float] = None,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None
    ) -> Optional[str]:
        try:
            img = Image.open(filepath)

            if rotation:
                img = img.rotate(-rotation, expand=True)
            if brightness:
                img = ImageEnhance.Brightness(img).enhance(brightness)
            if contrast:
                img = ImageEnhance.Contrast(img).enhance(contrast)

            base, ext = os.path.splitext(filepath)
            edited_filepath = f"{base}_edited{ext}"

            img.save(edited_filepath)
            return edited_filepath

        except Exception as e:
            logger.error(f"Error applying filter to {filepath}: {e}")
            return None

    def delete_media(self, filename: str) -> Tuple[bool, str]:
        media_path = os.path.join(self.upload_folder, filename)

        if os.path.exists(media_path):
            try:
                os.remove(media_path)
                logger.info(f"Deleted media: {filename}")

                dng_file = os.path.splitext(filename)[0] + '.dng'
                if os.path.exists(os.path.join(self.upload_folder, dng_file)):
                    os.remove(os.path.join(self.upload_folder, dng_file))
                    logger.info(f"Deleted corresponding DNG file: {dng_file}")

                return True, f"Media '{filename}' deleted successfully."
            except Exception as e:
                logger.error(f"Error deleting media {filename}: {e}")
                return False, "Failed to delete media"
        else:
            return False, "Media not found"

    def save_edit(
        self,
        filename: str,
        edits: Dict[str, Any],
        save_option: str,
        new_filename: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Apply edits to an image and save it based on user selection."""
        image_path = os.path.join(self.upload_folder, filename)
        logger.debug(f"Applying edits to {filename}: {edits}")

        if not os.path.exists(image_path):
            return False, "Original image not found."

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = ImageOps.exif_transpose(img)

                if "brightness" in edits:
                    brightness_factor = max(0.1, float(edits["brightness"]) / 100)
                    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

                if "contrast" in edits:
                    contrast_factor = max(0.1, float(edits["contrast"]) / 100)
                    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

                if "rotation" in edits:
                    rotation_angle = int(edits["rotation"]) % 360
                    rotation_angle = -rotation_angle
                    img = img.rotate(rotation_angle, expand=True)
                    logger.debug(f"Applied rotation: {rotation_angle}°")

                if save_option == "replace":
                    save_path = image_path
                elif save_option == "new_file" and new_filename:
                    save_path = os.path.join(self.upload_folder, new_filename)
                else:
                    return False, "Invalid save option."

                img.save(save_path)
                return True, "Image saved successfully."

        except Exception as e:
            logger.error(f"Error applying edits to image {filename}: {e}")
            return False, "Failed to edit image."