"""
Extracción de keypoints usando YOLOv8-pose.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# COCO keypoint indices (17-keypoint model)
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


class PoseExtractor:
    """Wraps a YOLO pose model to extract 2D keypoints from images/video."""

    def __init__(self, model_name: str = "yolov8n-pose.pt") -> None:
        self.model = YOLO(model_name)

    @staticmethod
    def _parse_keypoints(results, person_idx: int = 0) -> list[dict]:
        """
        Convert raw YOLO output into a list of {id, x, y, confidence} dicts
        for a specific person.

        Parameters
        ----------
        results : ultralytics Results
            YOLO prediction results.
        person_idx : int
            Index of the person to extract (default: 0).

        Returns
        -------
        list[dict]
            Empty list if no detections or person_idx out of range.
        """
        kps_data: list[dict] = []
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return kps_data

        n_people = len(results[0].keypoints.xy)
        if person_idx >= n_people:
            return kps_data

        kps = results[0].keypoints
        for i in range(len(kps.xy[person_idx])):
            x, y = kps.xy[person_idx][i].tolist()
            conf = float(kps.conf[person_idx][i].item())
            kps_data.append({"id": i, "x": x, "y": y, "confidence": conf})
        return kps_data

    def get_detections(self, frame: np.ndarray) -> list[dict]:
        """
        Get ALL people detected in a frame.

        Returns
        -------
        list[dict]
            Each dict::
                {
                    "keypoints": list of {id, x, y, confidence},
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float  # overall detection confidence
                }
            Empty list if no one detected.
        """
        results = self.model(frame, verbose=False)
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return []

        kps = results[0].keypoints
        boxes = results[0].boxes

        detections: list[dict] = []
        n_people = len(kps.xy)
        for p_idx in range(n_people):
            kp_list = []
            for i in range(len(kps.xy[p_idx])):
                x, y = kps.xy[p_idx][i].tolist()
                conf = float(kps.conf[p_idx][i].item())
                kp_list.append({"id": i, "x": x, "y": y, "confidence": conf})

            bbox = boxes.xyxy[p_idx].tolist() if boxes is not None else None
            det_conf = float(boxes.conf[p_idx].item()) if boxes is not None else 0.0

            detections.append({
                "keypoints": kp_list,
                "bbox": bbox,
                "confidence": det_conf,
            })

        return detections

    def extract_from_frame(
        self, frame: np.ndarray, person_idx: int = 0
    ) -> tuple[list[dict], np.ndarray]:
        """
        Run pose estimation on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame (BGR).
        person_idx : int
            Which person to extract (default: 0, the first detected).

        Returns
        -------
        keypoints : list[dict]
            List of {id, x, y, confidence} for every detected keypoint
            of the selected person.
        annotated : np.ndarray
            Frame with skeleton + keypoints drawn for ALL detected people.
        """
        results = self.model(frame, verbose=False)
        kps = self._parse_keypoints(results, person_idx=person_idx)
        annotated = results[0].plot()
        return kps, annotated

    def extract_from_frame_raw(self, frame: np.ndarray) -> tuple:
        """
        Like extract_from_frame but returns the full YOLO result.
        Useful when you need the keypoints object for advanced use.
        """
        results = self.model(frame, verbose=False)
        kps = self._parse_keypoints(results)
        return kps, results

    def process_video(
        self,
        video_path: str,
        frame_skip: int = 1,
        progress_callback=None,
    ) -> tuple[list[dict], str]:
        """
        Process an entire video file.

        Parameters
        ----------
        video_path : str
            Path to input video.
        frame_skip : int
            Process every Nth frame (1 = every frame).
        progress_callback : callable, optional
            Called with (frame_num, total_frames) for progress reporting.

        Returns
        -------
        frame_keypoints : list[dict]
            Each element: {frame: int, keypoints: list[dict]}.
        output_video_path : str
            Path to the annotated video file.
        """
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video
        output_video_path = video_path.rsplit(".", 1)[0] + "_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_num = 0
        frame_keypoints: list[dict] = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_skip == 0:
                kps, annotated = self.extract_from_frame(frame)
                frame_keypoints.append({"frame": frame_num, "keypoints": kps})
            else:
                annotated = frame

            writer.write(annotated)

            if progress_callback:
                progress_callback(frame_num + 1, total_frames)

            frame_num += 1

        cap.release()
        writer.release()
        return frame_keypoints, output_video_path

    def keypoints_to_dict(self, kps_list: list[dict]) -> dict[int, dict]:
        """Convert a list of {id, x, y, confidence} to {id: {x, y, confidence}}."""
        return {kp["id"]: {"x": kp["x"], "y": kp["y"], "confidence": kp["confidence"]} for kp in kps_list}

    # -----------------------------------------------------------------------
    # Multi-person tracking
    # -----------------------------------------------------------------------
    @staticmethod
    def _bbox_center(bbox: list[float]) -> tuple[float, float]:
        """Centro (x, y) de un bbox [x1, y1, x2, y2]."""
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _bbox_area(bbox: list[float]) -> float:
        """Area de un bbox [x1, y1, x2, y2]."""
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    @staticmethod
    def _greedy_match(current_centers, prev_centers, max_dist):
        """Empareja detecciones actuales con tracks previos por proximidad."""
        assignments = []
        used_prev: set[int] = set()
        used_curr: set[int] = set()
        for det_idx, c_center in enumerate(current_centers):
            best_dist = max_dist
            best_prev = None
            for prev_idx, p_center in enumerate(prev_centers):
                if prev_idx in used_prev:
                    continue
                dist = float(np.linalg.norm(np.array(c_center) - np.array(p_center)))
                if dist < best_dist:
                    best_dist = dist
                    best_prev = prev_idx
            if best_prev is not None:
                assignments.append((best_prev, det_idx))
                used_prev.add(best_prev)
                used_curr.add(det_idx)
        unmatched = [i for i in range(len(current_centers)) if i not in used_curr]
        return assignments, unmatched

    @staticmethod
    def _crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Recorte BGR de la region del bbox (miniatura del selector)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return frame.copy()
        return frame[y1:y2, x1:x2].copy()

    def _append_to_track(self, tracks, tid, det, frame, frame_num):
        """Agregar una deteccion a un track (creandolo si no existe)."""
        bbox = det["bbox"]
        area = self._bbox_area(bbox)
        if tid not in tracks:
            tracks[tid] = {
                "track_id": tid,
                "keypoints_per_frame": [],
                "n_frames": 0,
                "bbox": bbox,
                "max_area": area,
                "thumb": self._crop(frame, bbox),
            }
        tracks[tid]["keypoints_per_frame"].append(
            {"frame": frame_num, "kps": det["kps"]}
        )
        tracks[tid]["n_frames"] += 1
        if area >= tracks[tid]["max_area"]:
            tracks[tid]["max_area"] = area
            tracks[tid]["bbox"] = bbox
            tracks[tid]["thumb"] = self._crop(frame, bbox)

    def process_video_tracks(
        self,
        video_path: str,
        frame_skip: int = 3,
        conf_thresh: float = 0.5,
        track_dist: float = 200.0,
        min_track_frames: int = 3,
        progress_callback=None,
    ) -> tuple[list[dict], str]:
        """
        Procesa un video detectando y rastreando a TODAS las personas.

        Cada persona se sigue entre frames por proximidad del centro de su
        bounding box, conservando su identidad ("track") a lo largo del video.

        Returns
        -------
        tracks : list[dict]
            Ordenados de mayor a menor tamano (el primero = persona principal).
            Cada track: {track_id, keypoints_per_frame, n_frames, bbox,
            max_area, thumb}.
        annotated_video_path : str
            Video con esqueletos de todas las personas y cajas numeradas.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = video_path.rsplit(".", 1)[0] + "_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w, frame_h))

        tracks: dict[int, dict] = {}
        prev_centers: dict[int, tuple] = {}
        next_track_id = 0
        frame_num = 0
        palette = [
            (0, 200, 0), (0, 140, 255), (255, 80, 0),
            (200, 0, 200), (0, 220, 220), (180, 180, 0),
        ]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_skip != 0:
                writer.write(frame)
                frame_num += 1
                continue

            results = self.model(frame, verbose=False)
            annotated = results[0].plot()

            no_people = (
                results[0].keypoints is None
                or len(results[0].keypoints.xy) == 0
                or results[0].boxes is None
            )
            if no_people:
                writer.write(annotated)
                if progress_callback:
                    progress_callback(frame_num + 1, total_frames)
                frame_num += 1
                continue

            kps_all = results[0].keypoints
            current_dets = []
            current_centers = []
            n_people = len(kps_all.xy)
            for p_idx in range(n_people):
                bbox = results[0].boxes.xyxy[p_idx].tolist()
                det_conf = float(results[0].boxes.conf[p_idx].item())
                if det_conf < conf_thresh:
                    continue
                kps_dict = {}
                for i in range(len(kps_all.xy[p_idx])):
                    x, y = kps_all.xy[p_idx][i].tolist()
                    kps_dict[i] = {"x": x, "y": y}
                current_dets.append({"kps": kps_dict, "bbox": bbox})
                current_centers.append(self._bbox_center(bbox))

            prev_list = list(prev_centers.items())
            prev_ids = [p[0] for p in prev_list]
            prev_cent = [p[1] for p in prev_list]
            frame_assign: list[tuple[int, int]] = []

            if not prev_cent:
                for det_idx, det in enumerate(current_dets):
                    tid = next_track_id
                    next_track_id += 1
                    self._append_to_track(tracks, tid, det, frame, frame_num)
                    prev_centers[tid] = current_centers[det_idx]
                    frame_assign.append((tid, det_idx))
            else:
                assignments, unmatched = self._greedy_match(
                    current_centers, prev_cent, track_dist
                )
                new_centers = {}
                for prev_idx, det_idx in assignments:
                    tid = prev_ids[prev_idx]
                    self._append_to_track(
                        tracks, tid, current_dets[det_idx], frame, frame_num
                    )
                    new_centers[tid] = current_centers[det_idx]
                    frame_assign.append((tid, det_idx))
                for det_idx in unmatched:
                    tid = next_track_id
                    next_track_id += 1
                    self._append_to_track(
                        tracks, tid, current_dets[det_idx], frame, frame_num
                    )
                    new_centers[tid] = current_centers[det_idx]
                    frame_assign.append((tid, det_idx))
                prev_centers = new_centers

            for tid, det_idx in frame_assign:
                bbox = current_dets[det_idx]["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = palette[tid % len(palette)]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated, f"Persona {tid}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

            writer.write(annotated)
            if progress_callback:
                progress_callback(frame_num + 1, total_frames)
            frame_num += 1

        cap.release()
        writer.release()

        valid = [
            t for t in tracks.values()
            if len(t["keypoints_per_frame"]) >= min_track_frames
        ]
        valid.sort(key=lambda t: t["max_area"], reverse=True)
        return valid, out_path


def kps_to_array(kps_dict: dict[int, dict]) -> np.ndarray:
    """Convert keypoints dict to array of shape (17, 3) — x, y, confidence."""
    arr = np.zeros((17, 3), dtype=np.float32)
    for kid, v in kps_dict.items():
        arr[kid] = [v["x"], v["y"], v["confidence"]]
    return arr
