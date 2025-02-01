import logging
import os

import cv2
from moviepy import VideoFileClip
from yt_dlp import YoutubeDL


def create_video_from_images(images_path, video_output, fps=30):
    images = [img for img in os.listdir(images_path) if
              img.endswith(".png")]
    images.sort()
    first_image_path = os.path.join(images_path, images[0])
    frame = cv2.imread(str(first_image_path))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
    for img in images:
        image_path = os.path.join(images_path, img)
        frame = cv2.imread(image_path)
        video.write(frame)
    video.release()
    logging.info(f"Video created at {video_output}")


def video_to_frames(video_path, frames_path):
    if not os.path.exists(video_path):
        os.makedirs(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = os.path.join(
                frames_path,
                f"frame_{frame_count:04d}.png"
            )
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        cap.release()
        logging.info(f"Extracted {frame_count} frames to {frames_path}")


def download_video(url, output_path):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s')
    }
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info_dict)
    return video_path


def get_video_frames(video_path):
    return_value = {}
    try:
        cap = cv2.VideoCapture(video_path)
        return_value["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return_value["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
        return_value["duration"] = f"{return_value['frames'] / return_value['fps']:.2f}s"
    except Exception as e:
        error_message = f"Error while reading video file: {e}"
        logging.error(error_message)
        return_value = {
            "error": error_message
        }
    return return_value

def extract_audio(video_path, audio_path):
    return_value = None
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
        audio.close()
        video.close()
    except Exception as e:
        error_message = f"Error while extracting audio: {e}"
        logging.error(error_message)
        return_value = {
            "error": error_message
        }
    return return_value

