import json
import os
import logging
import subprocess
from datetime import datetime

# Image handeling imports
from PIL import Image, ImageOps, ImageEnhance

####################
# MediaGallery Class
####################

class MediaGallery:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.image_exts = ('.jpg', '.jpeg')
        self.video_exts = ('.mp4',)

    def get_image_resolution(self, path):
        with Image.open(path) as img:
            width, height = img.size
        return width, height

    def get_video_resolution(self, path):
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
            logging.warning(f"Could not read video resolution for {path}: {e}")
            return None, None

    def get_media_files(self, type="all"):
        try:
            files = os.listdir(self.upload_folder)
            media = []

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if type == "all" and ext not in self.image_exts + self.video_exts:
                    continue
                elif type == "image" and ext not in self.image_exts:
                    continue
                elif type == "video" and ext not in self.video_exts:
                    continue
                elif type not in ["all", "image", "video"]:
                    continue

                # Extract timestamp from filename
                try:
                    unix_ts = int(f.split('_')[-1].split('.')[0])
                    timestamp = datetime.utcfromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    continue

                path = os.path.join(self.upload_folder, f)
                item = {
                    "filename": f,
                    "timestamp": timestamp,
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

            media.sort(key=lambda x: x["timestamp"], reverse=True)
            return media

        except Exception as e:
            logging.error(f"Media loading error: {e}")
            return []

    def get_media_slice(self, offset=0, limit=20, type="all"):
        """Return a slice of media for infinite scroll."""
        all_media = self.get_media_files(type=type)
        sliced_media = all_media[offset:offset + limit]
        return sliced_media

    def find_last_image_taken(self):
        """Find the most recent image taken."""
        all_images = self.get_media_files(type="image")
        
        if all_images:
            first_image = all_images[0]
            print(f"Filename: {first_image['filename']}")
            image = first_image['filename']
        else:
            print("No image files found.")
            image = None
        
        return image  # Extract only the filename

    def apply_filter(self, filepath, rotation=None, brightness=None, contrast=None):
        try:
            img = Image.open(filepath)

            if rotation:
                img = img.rotate(-rotation, expand=True)
            if brightness:
                img = ImageEnhance.Brightness(img).enhance(brightness)
            if contrast:
                img = ImageEnhance.Contrast(img).enhance(contrast)

            # spilt fileextension from path
            base, ext = os.path.splitext(filepath)
            edited_filepath = f"{base}_edited{ext}"

            img.save(edited_filepath)
            return edited_filepath

        except Exception as e:
            print(f"Error applying filter: {e}")
            return None

    def delete_media(self, filename):
        media_path = os.path.join(self.upload_folder, filename)

        if os.path.exists(media_path):
            try:
                os.remove(media_path)
                logging.info(f"Deleted media: {filename}")
                # Check if corresponding .dng file exists
                dng_file = os.path.splitext(filename)[0] + '.dng'
                # print(dng_file)
                has_dng = os.path.exists(os.path.join(self.upload_folder, dng_file))
                # print(has_dng)
                if has_dng:
                    os.remove(os.path.join(self.upload_folder, dng_file))
                return True, f"Media '{filename}' deleted successfully."
            except Exception as e:
                logging.error(f"Error deleting media {filename}: {e}")
                return False, "Failed to delete media"
        else:
            return False, "Media not found"
    
    def save_edit(self, filename, edits, save_option, new_filename=None):
        """Apply edits to an image and save it based on user selection."""
        image_path = os.path.join(self.upload_folder, filename)
        print(f"Applying edits to {filename}: {edits}")

        if not os.path.exists(image_path):
            return False, "Original image not found."

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # Ensure no transparency issues

                # Reset EXIF rotation before applying new rotation
                img = ImageOps.exif_transpose(img)

                # Convert brightness and contrast from 0-200 range to 0.1-2.0
                if "brightness" in edits:
                    brightness_factor = max(0.1, float(edits["brightness"]) / 100)
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(brightness_factor)

                if "contrast" in edits:
                    contrast_factor = max(0.1, float(edits["contrast"]) / 100)
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(contrast_factor)

                # Apply absolute rotation (mod 360 to prevent stacking errors)
                if "rotation" in edits:
                    rotation_angle = int(edits["rotation"]) % 360
                    # Automatically convert the rotation to negative
                    rotation_angle = -rotation_angle
                    img = img.rotate(rotation_angle, expand=True)
                    print(f"Applied rotation: {rotation_angle}°")

                # Determine save path
                if save_option == "replace":
                    save_path = image_path
                elif save_option == "new_file" and new_filename:
                    save_path = os.path.join(self.upload_folder, new_filename)
                else:
                    return False, "Invalid save option."

                img.save(save_path)
                return True, "Image saved successfully."

        except Exception as e:
            logging.error(f"Error applying edits to image {filename}: {e}")
            return False, "Failed to edit image."
