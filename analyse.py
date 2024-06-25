import cv2

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration_seconds = total_frames / fps
    duration_minutes = duration_seconds / 60
    duration_hours = duration_minutes / 60
    cap.release()
    return total_frames, fps, duration_seconds, duration_minutes, duration_hours

video_path = 'a.mp4'  # Hier den Pfad zu Ihrer MP4-Datei angeben
frame_count, fps, duration_seconds, duration_minutes, duration_hours = get_video_info(video_path)

print(f"Das Video hat {frame_count} Frames.")
print(f"Die FPS des Videos sind {fps}.")
print(f"Die Gesamtdauer des Videos beträgt {duration_seconds:.2f} Sekunden.")
print(f"Das entspricht {duration_minutes:.2f} Minuten oder {duration_hours:.2f} Stunden.")
