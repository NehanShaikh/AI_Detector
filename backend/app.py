
import os
import math
import tempfile
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import pipeline


# ============================================================
# Configuration
# ============================================================

APP_TITLE = "AI Media Forensics API"
APP_VERSION = "0.8.0"

MODEL_ID = os.getenv(
    "MODEL_ID",
    "umm-maybe/AI-image-detector",
)

FACE_MODEL_ID = os.getenv(
    "FACE_MODEL_ID",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
)

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = 80 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
MAX_VIDEO_DURATION_SECONDS = 600
MAX_VIDEO_DIMENSION = 8192

# Heuristic decision threshold requested for the demo. This is not a validated
# probability of authenticity; calibrate it against a labeled test dataset.
VERDICT_THRESHOLD = 0.65

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/x-m4v",
}

IMAGE_MODEL_WARNING = (
    "The overall image classifier estimates whether an image "
    "resembles its learned classes. It does not prove authenticity."
)

FACE_MODEL_WARNING = (
    "The face model provides an estimate for a detected face crop. "
    "It is not proof that a face is genuine or manipulated. "
    "Its performance must be validated on representative data."
)

VIDEO_WARNING = (
    "This endpoint analyzes sampled frames, detects candidate faces, associates "
    "nearby face boxes across sampled frames, and summarizes score changes. "
    "This is heuristic temporal consistency analysis, not a dedicated or validated "
    "video deepfake detector. Model outputs are not proof of authenticity."
)


# ============================================================
# FastAPI application and CORS
# ============================================================

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
)

allowed_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip().rstrip("/")
        for origin in allowed_origins
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# Model caches
# ============================================================

_classifier = None
_model_error = None

_face_classifier = None
_face_model_error = None


def get_classifier():
    """Load and cache the general image classifier."""
    global _classifier, _model_error

    if _classifier is not None:
        return _classifier

    if _model_error:
        raise RuntimeError(_model_error)

    try:
        _classifier = pipeline(
            "image-classification",
            model=MODEL_ID,
            device=-1,
        )
        return _classifier

    except Exception as exc:
        _model_error = (
            f"Could not load image model '{MODEL_ID}'. "
            f"Check the model ID, internet connection, dependencies, "
            f"available disk space, and RAM. "
            f"Details: {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(_model_error) from exc


def get_face_classifier():
    """Load and cache the face classifier."""
    global _face_classifier, _face_model_error

    if _face_classifier is not None:
        return _face_classifier

    if _face_model_error:
        raise RuntimeError(_face_model_error)

    try:
        _face_classifier = pipeline(
            "image-classification",
            model=FACE_MODEL_ID,
            device=-1,
        )
        return _face_classifier

    except Exception as exc:
        _face_model_error = (
            f"Could not load face model '{FACE_MODEL_ID}'. "
            f"Check the model ID, compatibility, internet connection, "
            f"available disk space, and RAM. "
            f"Details: {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(_face_model_error) from exc


# ============================================================
# Classification helpers
# ============================================================

def run_classifier(
    classifier,
    image: Image.Image,
    model_id: str,
) -> dict[str, Any]:
    """Run a model and normalize its predictions."""

    raw_predictions = classifier(
        image.convert("RGB"),
        top_k=None,
    )

    if isinstance(raw_predictions, dict):
        raw_predictions = [raw_predictions]

    predictions = sorted(
        raw_predictions or [],
        key=lambda item: float(item.get("score", 0)),
        reverse=True,
    )

    if not predictions:
        raise RuntimeError(
            f"Model '{model_id}' returned no predictions."
        )

    top = predictions[0]

    return {
        "top_label": str(top.get("label", "unknown")),
        "top_score": round(float(top.get("score", 0)), 5),
        "predictions": [
            {
                "label": str(item.get("label", "unknown")),
                "score": round(float(item.get("score", 0)), 5),
            }
            for item in predictions[:10]
        ],
        "model_id": model_id,
    }


def classify_image(image: Image.Image) -> dict[str, Any]:
    return run_classifier(
        get_classifier(),
        image,
        MODEL_ID,
    )


def assess_label(label: str, model_type: str) -> str:
    """
    Interpret the model's class label without treating it as proof.
    """

    normalized = (
        label.strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )
    normalized = " ".join(normalized.split())

    if model_type == "face":
        if normalized in {"deepfake", "deep fake"}:
            return "model_predicts_deepfake"

        if normalized == "realism":
            return "model_predicts_realism"

        return "unmapped_label"

    if model_type == "image":
        if normalized in {
            "artificial",
            "ai generated",
            "synthetic",
            "generated",
        }:
            return "model_suggests_ai_generated"

        if normalized in {
            "human",
            "authentic",
            "natural",
            "real",
        }:
            return "model_suggests_human_or_real"

        return "unmapped_label"

    return "unmapped_label"


def get_prediction_score(
    classification: dict[str, Any],
    accepted_labels: set[str],
) -> float | None:
    """Return the score for a known class, if present."""

    for item in classification.get("predictions", []):
        label = str(item.get("label", "")).strip().lower()

        if label in accepted_labels:
            try:
                return float(item["score"])
            except (TypeError, ValueError, KeyError):
                return None

    return None


# ============================================================
# Face detection
# ============================================================

def find_faces(
    image: Image.Image,
) -> list[dict[str, int]]:
    """
    Detect candidate faces using OpenCV Haar cascades.

    Detection only locates candidate face regions. It does not
    determine whether a face is genuine or manipulated.
    """

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        raise RuntimeError(
            "OpenCV could not load its frontal-face Haar cascade."
        )

    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )

    return [
        {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
        }
        for x, y, w, h in boxes[:10]
    ]


# ============================================================
# Face crop classification
# ============================================================

def classify_face_crops(
    image: Image.Image,
    boxes: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """Classify detected face crops independently."""

    if not boxes:
        return []

    try:
        classifier = get_face_classifier()

    except Exception as exc:
        return [
            {
                "region": "face_model",
                "error": f"{type(exc).__name__}: {exc}",
                "warning": (
                    "Face classification could not run. "
                    "The overall image result is separate."
                ),
            }
        ]

    results = []

    for index, box in enumerate(boxes):
        x = box["x"]
        y = box["y"]
        w = box["width"]
        h = box["height"]

        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)

        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(image.width, x + w + margin_x)
        bottom = min(image.height, y + h + margin_y)

        crop = image.crop(
            (left, top, right, bottom)
        ).convert("RGB")

        try:
            classification = run_classifier(
                classifier,
                crop,
                FACE_MODEL_ID,
            )

            results.append(
                {
                    "region": f"face_{index + 1}",
                    "box": box,
                    "classification": classification,
                    "interpretation": assess_label(
                        classification["top_label"],
                        "face",
                    ),
                    "warning": FACE_MODEL_WARNING,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "region": f"face_{index + 1}",
                    "box": box,
                    "error": f"{type(exc).__name__}: {exc}",
                    "warning": (
                        "This face could not be classified. "
                        "No conclusion should be drawn for this region."
                    ),
                }
            )

    return results


# ============================================================
# Upload validation
# ============================================================

async def read_upload(
    file: UploadFile,
    allowed_types: set[str],
    max_bytes: int,
    media_description: str,
) -> bytes:
    """Validate declared MIME type and bound the upload size."""

    content_type = (
        file.content_type or ""
    ).split(";")[0].strip().lower()

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported {media_description} file type. "
                f"Received: {content_type or 'unknown'}."
            ),
        )

    data = await file.read(max_bytes + 1)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{media_description.capitalize()} is too large. "
                f"Maximum size is {max_bytes // (1024 * 1024)} MB."
            ),
        )

    return data


def safe_filename(filename: str | None) -> str:
    """Return a filename without directory components."""

    return Path(filename or "upload").name


def build_report_metadata(media_type: str, filename: str | None) -> dict[str, Any]:
    """Metadata to make an exported report easier to identify and audit."""
    return {
        "report_version": "1.0",
        "project_phase": "Phase 5 - Forensic Report and Evidence Visualization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "media_type": media_type,
        "filename": safe_filename(filename),
        "verdict_threshold": VERDICT_THRESHOLD,
        "verdict_is_heuristic": True,
    }


def build_video_evidence_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Count noteworthy model signals without claiming they prove manipulation."""
    threshold = VERDICT_THRESHOLD
    elevated_frame_signals = 0
    elevated_face_signals = 0
    frames_with_faces = 0
    failed_frames = 0
    evidence_items = []

    artificial_labels = {"artificial", "ai generated", "synthetic", "generated"}
    deepfake_labels = {"deepfake", "deep fake"}

    for item in results:
        frame_index = item.get("frame_index")
        timestamp = item.get("timestamp_seconds")
        if item.get("error"):
            failed_frames += 1
            continue

        frame_signal = None
        classification = item.get("classification") or {}
        for prediction in classification.get("predictions", []):
            label = str(prediction.get("label", "")).strip().lower().replace("_", " ").replace("-", " ")
            if label in artificial_labels:
                try:
                    frame_signal = float(prediction.get("score"))
                except (TypeError, ValueError):
                    pass
                break
        if frame_signal is not None and frame_signal >= threshold:
            elevated_frame_signals += 1
            evidence_items.append({
                "type": "frame_model_signal",
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "score": round(frame_signal, 5),
                "threshold": threshold,
                "description": "Artificial-class score crossed the heuristic threshold.",
            })

        faces = item.get("face_results") or []
        valid_faces = [face for face in faces if face.get("box")]
        if valid_faces:
            frames_with_faces += 1
        for face in valid_faces:
            face_classification = face.get("classification") or {}
            deepfake_score = None
            for prediction in face_classification.get("predictions", []):
                label = str(prediction.get("label", "")).strip().lower().replace("_", " ").replace("-", " ")
                if label in deepfake_labels:
                    try:
                        deepfake_score = float(prediction.get("score"))
                    except (TypeError, ValueError):
                        pass
                    break
            if deepfake_score is not None and deepfake_score >= threshold:
                elevated_face_signals += 1
                evidence_items.append({
                    "type": "face_model_signal",
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "track_id": face.get("track_id"),
                    "score": round(deepfake_score, 5),
                    "threshold": threshold,
                    "description": "Face-crop deepfake-class score crossed the heuristic threshold.",
                })

    return {
        "threshold": threshold,
        "frames_with_elevated_artificial_class_score": elevated_frame_signals,
        "face_regions_with_elevated_deepfake_class_score": elevated_face_signals,
        "frames_with_detected_faces": frames_with_faces,
        "failed_frames": failed_frames,
        "noteworthy_signals": evidence_items,
        "interpretation_note": (
            "Elevated signals identify model outputs for review; they are not proof of manipulation "
            "and are not calibrated probabilities of authenticity."
        ),
    }


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "phase": "Phase 5 - Forensic Report and Evidence Visualization",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model_loaded": _classifier is not None,
        "model_id": MODEL_ID,
        "model_error": _model_error,
        "face_model_loaded": _face_classifier is not None,
        "face_model_id": FACE_MODEL_ID,
        "face_model_error": _face_model_error,
        "video_analysis": {
            "enabled": True,
            "method": "sampled_frame_classification_face_crop_analysis_and_heuristic_tracking",
            "face_crop_analysis": True,
            "dedicated_video_model": False,
            "temporal_analysis": True,
            "temporal_method": "heuristic_face_track_score_consistency",
            "verdict_threshold": VERDICT_THRESHOLD,
            "verdict_is_heuristic": True,
        },
    }


# ============================================================
# Image analysis endpoint
# ============================================================

@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
):
    data = await read_upload(
        file,
        ALLOWED_IMAGE_TYPES,
        MAX_IMAGE_BYTES,
        "image",
    )

    try:
        with Image.open(BytesIO(data)) as source:
            width, height = source.size

            # Check dimensions before decoding all image pixels.
            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image dimensions.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large.",
                )

            source.load()
            image = source.convert("RGB")

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The file could not be decoded as an image.",
        )

    try:
        overall = classify_image(image)

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The overall image model could not analyze this image. "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    try:
        face_boxes = find_faces(image)
        face_results = classify_face_crops(image, face_boxes)

    except Exception as exc:
        face_boxes = []
        face_results = [
            {
                "region": "face_detection",
                "error": f"{type(exc).__name__}: {exc}",
                "warning": (
                    "Face detection or analysis failed. "
                    "The overall image result is separate."
                ),
            }
        ]

    return {
        "report_metadata": build_report_metadata("image", file.filename),
        "filename": safe_filename(file.filename),
        "media_type": "image",
        "dimensions": {
            "width": image.width,
            "height": image.height,
        },
        "overall": overall,
        "interpretation": assess_label(
            overall["top_label"],
            "image",
        ),
        "face_count": len(face_boxes),
        "face_regions": face_results,
        "assessment_policy": {
            "combined_verdict": False,
            "reason": (
                "The overall image classifier and face classifier "
                "are independent models with different purposes. "
                "Their scores are not combined into a single verdict."
            ),
        },
        "region_analysis_note": (
            "Face boxes are detected using OpenCV Haar cascades. "
            "The overall image model and face-crop model are separate. "
            "The face model has not been validated for every type of "
            "AI-generated face or face manipulation. No validated "
            "background-only manipulation detector is included."
        ),
        "warnings": [
            IMAGE_MODEL_WARNING,
            FACE_MODEL_WARNING,
            "Model scores are not guaranteed probabilities of authenticity.",
            "A face-crop prediction does not prove that a face was swapped.",
            "A real-looking prediction does not prove an image is genuine.",
            "A model prediction should not be used alone to accuse a person "
            "of manipulating an image.",
        ],
        "limitations": [
            "Face detection may miss faces or detect them inaccurately.",
            "Small or low-resolution faces may be difficult to classify.",
            "Performance may vary across datasets and manipulation methods.",
            "The overall image model may not detect every AI-generated image.",
            "The face model may not reliably detect fully AI-generated faces.",
            "Image analysis does not establish image provenance.",
        ],
    }


# ============================================================
# Video analysis helpers
# ============================================================

def get_video_metadata(cap: cv2.VideoCapture) -> dict[str, Any]:
    """Read and validate basic video metadata."""

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS) or 0
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    )

    if not math.isfinite(fps) or fps < 0:
        fps = 0.0

    duration = (
        frame_count / fps
        if frame_count > 0 and fps > 0
        else None
    )

    if duration is not None and not math.isfinite(duration):
        duration = None

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": (
            round(duration, 2)
            if duration is not None
            else None
        ),
    }


def summarize_video_frames(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize frame-model scores and produce a threshold-based signal."""

    label_counts: dict[str, int] = {}
    artificial_scores: list[float] = []
    human_scores: list[float] = []

    for item in results:
        classification = item.get("classification")
        if not classification:
            continue

        label = classification["top_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

        artificial_score = get_prediction_score(
            classification,
            {"artificial", "ai generated", "synthetic", "generated"},
        )
        human_score = get_prediction_score(
            classification,
            {"human", "real", "authentic", "natural"},
        )

        if artificial_score is not None:
            artificial_scores.append(artificial_score)
        if human_score is not None:
            human_scores.append(human_score)

    mean_artificial = (
        float(np.mean(artificial_scores)) if artificial_scores else None
    )
    mean_human = float(np.mean(human_scores)) if human_scores else None

    frame_signal = "inconclusive"
    frame_reason = (
        f"Neither mean frame class score reached the {VERDICT_THRESHOLD:.0%} threshold."
    )
    if mean_artificial is not None and mean_artificial >= VERDICT_THRESHOLD:
        frame_signal = "likely_fake"
        frame_reason = (
            "The mean artificial-associated frame score reached the configured "
            f"{VERDICT_THRESHOLD:.0%} heuristic threshold."
        )
    elif mean_human is not None and mean_human >= VERDICT_THRESHOLD:
        frame_signal = "likely_real"
        frame_reason = (
            "The mean human-associated frame score reached the configured "
            f"{VERDICT_THRESHOLD:.0%} heuristic threshold."
        )

    return {
        "analyzed_frames": len(results),
        "successful_frames": sum(1 for item in results if "classification" in item),
        "failed_frames": sum(1 for item in results if "error" in item),
        "label_counts": label_counts,
        "mean_artificial_class_score": (
            round(mean_artificial, 5) if mean_artificial is not None else None
        ),
        "mean_human_class_score": (
            round(mean_human, 5) if mean_human is not None else None
        ),
        "frame_signal_verdict": frame_signal,
        "frame_signal_reason": frame_reason,
        # The final verdict is assigned after face analysis is summarized.
        "verdict": "inconclusive",
        "verdict_reason": "Final verdict is determined from available model signals.",
        "verdict_threshold": VERDICT_THRESHOLD,
    }


def summarize_video_face_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize face-crop model scores and derive a cautious threshold signal."""
    frames_with_faces = 0
    detected_faces = 0
    classified_faces = 0
    failed_face_classifications = 0
    predicts_deepfake = 0
    predicts_realism = 0
    deepfake_scores: list[float] = []
    realism_scores: list[float] = []

    for frame_result in results:
        if "classification" not in frame_result:
            continue

        face_results = frame_result.get("face_results", [])
        valid_face_results = [
            item for item in face_results
            if item.get("region", "").startswith("face_")
        ]
        if frame_result.get("face_count", 0) > 0:
            frames_with_faces += 1
        detected_faces += int(frame_result.get("face_count", 0))

        for face in valid_face_results:
            classification = face.get("classification")
            if not classification:
                if "error" in face:
                    failed_face_classifications += 1
                continue

            classified_faces += 1
            interpretation = face.get("interpretation")
            if interpretation == "model_predicts_deepfake":
                predicts_deepfake += 1
            elif interpretation == "model_predicts_realism":
                predicts_realism += 1

            deepfake_score = get_prediction_score(classification, {"deepfake", "deep fake"})
            realism_score = get_prediction_score(classification, {"realism", "real"})
            if deepfake_score is not None:
                deepfake_scores.append(deepfake_score)
            if realism_score is not None:
                realism_scores.append(realism_score)

    mean_deepfake = float(np.mean(deepfake_scores)) if deepfake_scores else None
    mean_realism = float(np.mean(realism_scores)) if realism_scores else None

    face_signal = "inconclusive"
    face_reason = (
        "No usable face-model score reached the configured "
        f"{VERDICT_THRESHOLD:.0%} threshold."
    )
    if mean_deepfake is not None and mean_deepfake >= VERDICT_THRESHOLD:
        face_signal = "likely_fake"
        face_reason = (
            "The mean face-model Deepfake score reached the configured "
            f"{VERDICT_THRESHOLD:.0%} heuristic threshold."
        )
    elif mean_realism is not None and mean_realism >= VERDICT_THRESHOLD:
        face_signal = "likely_real"
        face_reason = (
            "The mean face-model Realism score reached the configured "
            f"{VERDICT_THRESHOLD:.0%} heuristic threshold."
        )

    return {
        "frames_with_detected_faces": frames_with_faces,
        "detected_face_regions": detected_faces,
        "successfully_classified_face_regions": classified_faces,
        "failed_face_classifications": failed_face_classifications,
        "face_model_predicts_deepfake_count": predicts_deepfake,
        "face_model_predicts_realism_count": predicts_realism,
        "mean_face_deepfake_score": round(mean_deepfake, 5) if mean_deepfake is not None else None,
        "mean_face_realism_score": round(mean_realism, 5) if mean_realism is not None else None,
        "face_signal_verdict": face_signal,
        "face_signal_reason": face_reason,
        "verdict_threshold": VERDICT_THRESHOLD,
        "note": (
            "This threshold-based signal is a heuristic, not a calibrated probability. "
            "The face model can produce false positives and has not been validated "
            "for this video domain."
        ),
    }




# ============================================================
# Phase 4: heuristic face tracking and temporal score analysis
# ============================================================

def _normalized_face_center(
    box: dict[str, int],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    """Return a face-box center normalized to frame dimensions."""
    cx = box["x"] + box["width"] / 2
    cy = box["y"] + box["height"] / 2
    return (
        cx / max(frame_width, 1),
        cy / max(frame_height, 1),
    )


def track_faces_across_sampled_frames(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Associate face boxes across sampled frames using normalized center distance.

    This is a lightweight heuristic, not biometric identity recognition. Large
    motion, camera movement, missed detections, or multiple people may break tracks.
    """
    tracks: dict[str, dict[str, Any]] = {}
    next_track_number = 1
    max_center_distance = 0.20
    max_sample_gap = 2

    for result in results:
        if "classification" not in result:
            continue

        dims = result.get("dimensions") or {}
        frame_width = int(dims.get("width") or 1)
        frame_height = int(dims.get("height") or 1)
        frame_number = int(result.get("frame_index", 0))
        timestamp = result.get("timestamp_seconds")
        used_tracks: set[str] = set()

        face_results = result.get("face_results", [])
        for face in face_results:
            region = str(face.get("region", ""))
            if not region.startswith("face_"):
                continue

            box = face.get("box")
            if not isinstance(box, dict):
                continue

            center = _normalized_face_center(box, frame_width, frame_height)
            best_track_id = None
            best_distance = float("inf")

            for track_id, track in tracks.items():
                if track_id in used_tracks:
                    continue
                if frame_number - track["last_frame_index"] > max_sample_gap * max(
                    1, int(result.get("sample_step_frames", 1))
                ):
                    # Skip stale tracks. The endpoint may not provide sample_step,
                    # so the distance criterion below remains the main safeguard.
                    continue

                previous_center = track["last_center"]
                distance = math.sqrt(
                    (center[0] - previous_center[0]) ** 2
                    + (center[1] - previous_center[1]) ** 2
                )
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None or best_distance > max_center_distance:
                best_track_id = f"face_track_{next_track_number}"
                next_track_number += 1
                tracks[best_track_id] = {
                    "track_id": best_track_id,
                    "observations": [],
                    "last_center": center,
                    "last_frame_index": frame_number,
                }

            track = tracks[best_track_id]
            used_tracks.add(best_track_id)
            track["last_center"] = center
            track["last_frame_index"] = frame_number

            face["track_id"] = best_track_id
            classification = face.get("classification")
            deepfake_score = (
                get_prediction_score(classification, {"deepfake", "deep fake"})
                if isinstance(classification, dict)
                else None
            )
            realism_score = (
                get_prediction_score(classification, {"realism", "real"})
                if isinstance(classification, dict)
                else None
            )
            track["observations"].append({
                "frame_index": frame_number,
                "timestamp_seconds": timestamp,
                "deepfake_score": deepfake_score,
                "realism_score": realism_score,
                "interpretation": face.get("interpretation"),
            })

    track_summaries = []
    all_score_changes = []
    unstable_tracks = 0

    for track_id, track in tracks.items():
        observations = track["observations"]
        deepfake_values = [
            float(obs["deepfake_score"])
            for obs in observations
            if obs["deepfake_score"] is not None
        ]
        realism_values = [
            float(obs["realism_score"])
            for obs in observations
            if obs["realism_score"] is not None
        ]

        score_std = (
            float(np.std(deepfake_values))
            if len(deepfake_values) >= 2
            else None
        )
        score_range = (
            max(deepfake_values) - min(deepfake_values)
            if len(deepfake_values) >= 2
            else None
        )
        if score_range is not None:
            all_score_changes.append(score_range)
        # This is a descriptive heuristic, not evidence of manipulation.
        unstable = score_std is not None and score_std >= 0.20
        if unstable:
            unstable_tracks += 1

        track_summaries.append({
            "track_id": track_id,
            "observations": len(observations),
            "frame_indices": [obs["frame_index"] for obs in observations],
            "timestamps_seconds": [obs["timestamp_seconds"] for obs in observations],
            "mean_deepfake_score": (
                round(float(np.mean(deepfake_values)), 5)
                if deepfake_values else None
            ),
            "mean_realism_score": (
                round(float(np.mean(realism_values)), 5)
                if realism_values else None
            ),
            "deepfake_score_range": (
                round(float(score_range), 5) if score_range is not None else None
            ),
            "deepfake_score_standard_deviation": (
                round(float(score_std), 5) if score_std is not None else None
            ),
            "score_variability_flag": unstable,
            "note": (
                "Track association and score variability are heuristic; "
                "they do not establish identity or prove manipulation."
            ),
        })

    repeated_tracks = sum(1 for item in track_summaries if item["observations"] >= 2)
    return {
        "tracking_method": "normalized_face_center_distance",
        "tracks_found": len(track_summaries),
        "tracks_seen_in_multiple_sampled_frames": repeated_tracks,
        "tracks_with_high_face_score_variability": unstable_tracks,
        "mean_deepfake_score_range_across_tracks": (
            round(float(np.mean(all_score_changes)), 5)
            if all_score_changes else None
        ),
        "tracks": track_summaries,
        "temporal_signal": "heuristic_score_variability_only",
        "note": (
            "This compares model scores for heuristically associated face boxes "
            "across sampled frames. It does not inspect optical flow, facial "
            "landmarks, lip-sync, frame-to-frame warping, or other validated "
            "temporal deepfake cues."
        ),
    }


def combine_video_signals(
    frame_summary: dict[str, Any],
    face_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Combine the independent threshold signals conservatively.

    Agreement can produce a heuristic likely-real/likely-fake label. Disagreement
    returns inconclusive rather than allowing one model to silently override another.
    """
    frame_signal = frame_summary.get("frame_signal_verdict", "inconclusive")
    face_signal = face_summary.get("face_signal_verdict", "inconclusive")
    usable_faces = int(face_summary.get("successfully_classified_face_regions", 0))

    if frame_signal in {"likely_real", "likely_fake"} and face_signal in {
        "likely_real", "likely_fake"
    }:
        if frame_signal == face_signal:
            verdict = frame_signal
            reason = (
                "The frame-model and face-model heuristic signals agree and each "
                f"crossed the {VERDICT_THRESHOLD:.0%} threshold. This is not a "
                "validated probability or proof of authenticity."
            )
            basis = "agreement_between_frame_and_face_models"
        else:
            verdict = "inconclusive"
            reason = (
                "The frame-model and face-model signals disagree. The system "
                "does not force a real/fake verdict when its models conflict."
            )
            basis = "conflicting_model_signals"
    elif usable_faces > 0 and face_signal in {"likely_real", "likely_fake"}:
        verdict = face_signal
        reason = (
            "Only the face-model signal crossed the configured threshold. "
            "This is a heuristic model indication, not proof of authenticity."
        )
        basis = "face_crop_model_only"
    elif frame_signal in {"likely_real", "likely_fake"}:
        verdict = frame_signal
        reason = (
            "Only the sampled-frame model signal crossed the configured threshold. "
            "This is a heuristic model indication, not proof of authenticity."
        )
        basis = "sampled_frame_model_only"
    else:
        verdict = "inconclusive"
        reason = (
            "Neither model produced a usable threshold signal strong enough for "
            "the current heuristic rules."
        )
        basis = "no_consistent_threshold_signal"

    return {
        "verdict": verdict,
        "verdict_reason": reason,
        "final_verdict_basis": basis,
        "frame_signal_verdict": frame_signal,
        "face_signal_verdict": face_signal,
        "verdict_threshold": VERDICT_THRESHOLD,
        "combination_policy": (
            "Require agreement when both models produce threshold-crossing signals; "
            "if only one model crosses the threshold, report its heuristic signal; "
            "if signals conflict, return inconclusive."
        ),
    }


# ============================================================
# Video analysis endpoint - Phase 4
# ============================================================

@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    max_frames: int = Form(12),
):
    """
    Analyze sampled frames, classify detected face crops, associate face boxes
    across samples, summarize score variability, and combine model signals
    conservatively. This is not a validated video deepfake detector.
    """

    if not 1 <= max_frames <= 40:
        raise HTTPException(
            status_code=400,
            detail="max_frames must be between 1 and 40.",
        )
        
        

    data = await read_upload(
        file,
        ALLOWED_VIDEO_TYPES,
        MAX_VIDEO_BYTES,
        "video",
    )

    suffix = Path(
        file.filename or "upload.mp4"
    ).suffix.lower()

    allowed_suffixes = {
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".m4v",
    }

    if suffix not in allowed_suffixes:
        suffix = ".mp4"

    temp_path = None
    cap = None

    try:
        # The configured system temp directory is used.
        # Set TMP and TEMP in PowerShell if you need temporary
        # files stored on another drive.
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            suffix=suffix,
        ) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not open this video. "
                    "Try a standard MP4, MOV, AVI, or WebM file."
                ),
            )

        metadata = get_video_metadata(cap)

        frame_count = metadata["frame_count"]
        fps = metadata["fps"]
        width = metadata["width"]
        height = metadata["height"]
        duration = metadata["duration_seconds"]

        if frame_count <= 0:
            raise HTTPException(
                status_code=400,
                detail="Could not read frames from this video.",
            )

        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400,
                detail="The video reports invalid dimensions.",
            )

        if max(width, height) > MAX_VIDEO_DIMENSION:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Video dimensions exceed the supported limit "
                    f"of {MAX_VIDEO_DIMENSION} pixels per side."
                ),
            )

        if width * height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail="Video frame dimensions are too large.",
            )

        if (
            duration is not None
            and duration > MAX_VIDEO_DURATION_SECONDS
        ):
            raise HTTPException(
                status_code=413,
                detail=(
                    "Video duration exceeds the limit of "
                    f"{MAX_VIDEO_DURATION_SECONDS // 60} minutes."
                ),
            )

        sample_count = min(max_frames, frame_count)

        # Select evenly spaced frames, including the first and last
        # frame when more than one frame is sampled.
        indices = np.linspace(
            0,
            frame_count - 1,
            num=sample_count,
            dtype=int,
        )

        selected_indices = sorted(
            set(int(index) for index in indices)
        )

        results = []

        for frame_index in selected_indices:
            # Seek to a frame and read it.
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            ok, frame = cap.read()

            timestamp = (
                round(frame_index / fps, 3)
                if fps > 0
                else None
            )

            if not ok or frame is None:
                results.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "error": "Could not decode this sampled frame.",
                    }
                )
                continue

            frame_height, frame_width = frame.shape[:2]

            if (
                frame_width <= 0
                or frame_height <= 0
                or frame_width * frame_height > MAX_IMAGE_PIXELS
            ):
                results.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": timestamp,
                        "error": "Sampled frame dimensions are invalid or too large.",
                    }
                )
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                prediction = classify_image(pil_image)
            except Exception as exc:
                # A failed frame-level prediction is recorded explicitly.
                results.append({
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "error": f"Frame classification failed: {type(exc).__name__}: {exc}",
                })
                continue

            # Face analysis is separate from frame classification. A face-model
            # failure must not erase an otherwise successful frame prediction.
            face_boxes = []
            face_results = []
            face_detection_error = None
            try:
                face_boxes = find_faces(pil_image)[:5]
                face_results = classify_face_crops(pil_image, face_boxes)
            except Exception as exc:
                face_detection_error = f"{type(exc).__name__}: {exc}"
                face_results = [{
                    "region": "face_detection",
                    "error": face_detection_error,
                    "warning": (
                        "Face detection failed for this frame. The frame-level "
                        "classification is separate and remains available."
                    ),
                }]

            frame_result = {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "dimensions": {"width": frame_width, "height": frame_height},
                "classification": prediction,
                "interpretation": assess_label(prediction["top_label"], "image"),
                "face_count": len(face_boxes),
                "face_results": face_results,
                "face_detection_error": face_detection_error,
            }
            results.append(frame_result)

        successful_count = sum(
            1 for item in results if "classification" in item
        )

        if successful_count == 0:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No sampled frames could be classified. "
                    "Check the model status and video codec."
                ),
            )

        summary = summarize_video_frames(results)
        face_summary = summarize_video_face_results(results)

        temporal_summary = track_faces_across_sampled_frames(results)
        combined = combine_video_signals(summary, face_summary)
        summary.update(combined)

        return {
            "report_metadata": build_report_metadata("video", file.filename),
            "evidence_summary": build_video_evidence_summary(results),
            "filename": safe_filename(file.filename),
            "media_type": "video",
            "video_info": {
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "duration_seconds": duration,
                "reported_total_frames": frame_count,
                "requested_max_frames": max_frames,
                "sampled_frames": len(results),
            },
            "summary": summary,
            "face_analysis_summary": face_summary,
            "temporal_analysis_summary": temporal_summary,
            "frame_results": results,
            "analysis_method": {
                "type": "sampled_frame_classification_face_crop_analysis_and_heuristic_tracking",
                "model_id": MODEL_ID,
                "face_model_id": FACE_MODEL_ID,
                "dedicated_video_model": False,
                "temporal_analysis": True,
                "temporal_analysis_type": "heuristic_face_track_score_consistency",
                "face_crop_analysis": True,
                "face_tracking": True,
                "max_face_regions_per_frame": 5,
            },
            "warning": VIDEO_WARNING,
            "limitations": [
                "Only selected frames were analyzed.",
                "Manipulations between sampled frames may be missed.",
                "The general image classifier is not a validated video detector.",
                "Temporal analysis is limited to heuristic face-box association and model-score variability.",
                "No optical-flow, facial-landmark, lip-sync, audio-video synchronization, or dedicated temporal deepfake model is used.",
                "Heuristic face tracks may switch identities or lose track when faces move or detections fail.",
                "Face detection is limited to candidate regions found by an OpenCV Haar cascade.",
                "The face classifier is run on detected face crops, but has not been validated for this video domain.",
                "The 65% verdict threshold is a demo heuristic, not a calibrated probability of authenticity.",
                "A likely-real or likely-fake verdict can be wrong; validate on a labeled test set before relying on it.",
                "Frame predictions do not prove that a video or person is fake.",
                "Video compression, lighting, blur, and low resolution may affect results.",
            ],
        }

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                # Do not expose a local temporary path in the API response.
                pass

        await file.close()
        
