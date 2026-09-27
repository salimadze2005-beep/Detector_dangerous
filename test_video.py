import cv2
import logging
from core.event_bus import EventBus, Event
from video.fall_detector import FallDetector
from core.config import get_resource_path
import math

logging.basicConfig(level=logging.INFO)


def on_fall(event: Event):
    print(
        f"\n🚨 [EVENT BUS] FALL DETECTED! Track ID: {event.metadata.get('track_id')}, Video Duration: {event.metadata.get('duration_sec')}s\n")


def test_video(video_path: str, output_path: str = "output_alert.mp4"):
    bus = EventBus()
    bus.subscribe("FALL_DETECTED", on_fall)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return

    # Извлекаем оригинальный FPS из самого видеофайла
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = FallDetector(
        camera_id="manual-test-camera",
        model_path=get_resource_path("models/yolov8n-pose.pt"),
        fall_duration_sec=5.0,  # Ровно 5 секунд по видеоряду
        fps=fps,
        use_wall_clock=False,
    )

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing '{video_path}' (Detected FPS: {fps:.2f})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        analysis = detector.process_frame(frame)
        annotated_frame = analysis.annotated_frame
        for event in analysis.events:
            bus.publish(event)
        out.write(annotated_frame)

        display_frame = cv2.resize(annotated_frame, (1080, 720))
        cv2.imshow("SecurityAI", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    bus.flush()
    bus.stop()


if __name__ == "__main__":
    test_video("tests/2026-08-06 16-54-37.mp4", "tests/fall_output2.mp4")

