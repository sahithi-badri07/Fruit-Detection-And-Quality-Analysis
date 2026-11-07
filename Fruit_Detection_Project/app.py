import hmac
import hashlib
import re
import secrets
import sqlite3
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
TRAINED_MODEL = PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"
PRETRAINED_MODEL = PROJECT_DIR / "yolov8n.pt"
ACCOUNT_DB = Path.home() / ".fruit_quality_inspector" / "accounts.sqlite3"
MAX_VIDEO_SAMPLES = 60
PASSWORD_HASH_ITERATIONS = 310_000

QUALITY_LABELS = {
    "rotten": ("Rotten", (0, 0, 230)),
    "spoiled": ("Spoiled", (0, 0, 230)),
    "poor": ("Poor", (0, 0, 230)),
    "fresh": ("Fresh", (0, 200, 0)),
    "excellent": ("Excellent", (0, 200, 0)),
    "good": ("Good", (0, 200, 0)),
    "unripe": ("Unripe", (0, 210, 255)),
}


st.set_page_config(
    page_title="Fruit Quality Inspector",
    page_icon="🍎",
    layout="wide",
)


def initialize_accounts() -> None:
    ACCOUNT_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ACCOUNT_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_HASH_ITERATIONS
    )


def create_account(username: str, password: str) -> bool:
    salt = secrets.token_bytes(16)
    digest = password_hash(password, salt)
    try:
        with sqlite3.connect(ACCOUNT_DB) as connection:
            connection.execute(
                "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                (username, salt, digest),
            )
    except sqlite3.IntegrityError:
        return False
    return True


def verify_account(username: str, password: str) -> bool:
    with sqlite3.connect(ACCOUNT_DB) as connection:
        account = connection.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if account is None:
        return False
    salt, expected_hash = account
    return hmac.compare_digest(password_hash(password, salt), expected_hash)


def require_login() -> None:
    if st.session_state.get("authenticated", False):
        return

    try:
        initialize_accounts()
    except OSError as error:
        st.error(f"Could not initialize the local account database: {error}")
        st.stop()
    except sqlite3.Error as error:
        st.error(f"Could not initialize the local account database: {error}")
        st.stop()

    st.markdown(
        """
        <style>
        .block-container { max-width: 900px; padding-top: 4rem; }
        .login-title {
            color: #32678d;
            font-size: clamp(2rem, 5vw, 3.2rem);
            font-weight: 750;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
        <div class="login-title">Fruit Detection App — Login</div>
        """,
        unsafe_allow_html=True,
    )
    signup_tab, login_tab = st.tabs(["Sign Up", "Login"])
    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Create username")
            new_password = st.text_input("Create password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            signup_submitted = st.form_submit_button("Create Account")
        if signup_submitted:
            normalized_username = new_username.strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", normalized_username):
                st.error(
                    "Username must be 3–32 characters and contain only letters, "
                    "numbers, underscores, or hyphens."
                )
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    if create_account(normalized_username, new_password):
                        st.success("Account created. Select Login and sign in.")
                    else:
                        st.error("That username is already registered.")
                except sqlite3.Error as error:
                    st.error(f"Could not create the account: {error}")
    with login_tab:
        with st.form("login_form"):
            entered_username = st.text_input("Username")
            entered_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")
        if login_submitted:
            try:
                if verify_account(entered_username.strip(), entered_password):
                    st.session_state["authenticated"] = True
                    st.rerun()
                st.error("Incorrect username or password.")
            except sqlite3.Error as error:
                st.error(f"Could not verify the account: {error}")
    st.stop()


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def label_details(class_name: str) -> tuple[str, str | None, tuple[int, int, int]]:
    normalized = re.sub(r"[^a-z0-9]+", " ", class_name.casefold()).strip()
    words = normalized.split()
    quality = next(
        (
            (label, color)
            for key, (label, color) in QUALITY_LABELS.items()
            if key in words
        ),
        None,
    )
    fruit_name = " ".join(word for word in words if quality is None or word not in {
        key for key in QUALITY_LABELS if key in words
    })
    return (
        fruit_name.title() or class_name,
        quality[0] if quality else None,
        quality[1] if quality else (255, 170, 45),
    )


def predict_frame(frame: np.ndarray, model: YOLO, use_pretrained_model: bool):
    if use_pretrained_model:
        fruit_classes = [
            class_id
            for class_id, name in model.names.items()
            if name in {"apple", "banana", "orange"}
        ]
        result = model(frame, classes=fruit_classes, verbose=False)[0]
    else:
        result = model(frame, verbose=False)[0]

    annotated = frame.copy()
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = str(model.names[class_id])
        fruit_name, quality, color = label_details(class_name)
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = f"{fruit_name} {confidence:.2f}"
        if quality:
            label = f"{fruit_name} · {quality} {confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + text_width + 8, y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, y1 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        detections.append(
            {
                "name": fruit_name,
                "class_name": class_name,
                "quality": quality,
                "confidence": confidence,
            }
        )
    return annotated, detections


def show_summary(detections: list[dict], title: str = "Quality Summary") -> None:
    st.subheader(f"📊 {title}")
    st.metric("Total Fruits", len(detections))
    quality_detections = [item for item in detections if item["quality"]]
    if quality_detections:
        st.metric("Quality Labels", len(quality_detections))
        counts: dict[str, int] = {}
        for item in quality_detections:
            quality = item["quality"]
            counts[quality] = counts.get(quality, 0) + 1
        for quality, count in counts.items():
            st.write(f"**{quality}:** {count}")
        if any(item["quality"] in {"Rotten", "Spoiled", "Poor"} for item in quality_detections):
            st.warning("The model classified one or more fruits as poor quality.")
    else:
        st.info("Quality assessment is unavailable for this model.")


def show_details(detections: list[dict]) -> None:
    st.subheader("📋 Detailed Fruit Analysis")
    if not detections:
        st.info("No fruits detected in this input.")
        return
    for index, item in enumerate(detections, start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {item['name']}**")
            st.write(f"Confidence: {item['confidence']:.0%}")
            if item["quality"]:
                st.write(f"Quality: {item['quality']} (model class)")
            else:
                st.write("Quality: Not available for this model")


def show_image_result(uploaded_file, model: YOLO, use_pretrained_model: bool) -> None:
    image_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("The uploaded file could not be decoded as an image.")
        return

    annotated, detections = predict_frame(image, model, use_pretrained_model)
    left, right = st.columns([1.7, 1])
    with left:
        st.subheader("📸 Detected Fruits")
        st.image(annotated, channels="BGR", width="stretch")
        st.caption("Annotated with detection and available model quality labels.")
    with right:
        show_summary(detections)
    show_details(detections)


def show_video_result(uploaded_file, model: YOLO, use_pretrained_model: bool) -> None:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    temporary_path: Path | None = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(uploaded_file.getvalue())
            temporary_path = Path(temporary_file.name)
        capture = cv2.VideoCapture(str(temporary_path))
        if not capture.isOpened():
            st.error("The uploaded video could not be opened.")
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_step = max(1, round(fps)) if np.isfinite(fps) and fps > 0 else 1
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frames = []
        all_detections = []
        frame_index = 0
        while len(sampled_frames) < MAX_VIDEO_SAMPLES:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_step == 0:
                annotated, detections = predict_frame(frame, model, use_pretrained_model)
                sampled_frames.append((frame_index, annotated))
                all_detections.extend(detections)
            frame_index += 1
        if not sampled_frames:
            st.error("No readable frames were found in the uploaded video.")
            return

        st.video(uploaded_file.getvalue())
        if total_frames > 0 and total_frames / frame_step > MAX_VIDEO_SAMPLES:
            st.info("Analysis is limited to the first 60 one-second samples.")
        left, right = st.columns([1.7, 1])
        with left:
            st.subheader("🎞️ Detected Fruits")
            for sample_index, (frame_number, annotated) in enumerate(sampled_frames[:3], start=1):
                st.image(
                    annotated,
                    channels="BGR",
                    width="stretch",
                    caption=f"Sample {sample_index} · frame {frame_number}",
                )
            st.caption(
                f"Showing up to 3 of {len(sampled_frames)} analyzed frames "
                f"(approximately one frame per second)."
            )
        with right:
            show_summary(all_detections, "Video Summary")
            st.caption("Counts are detections across sampled frames, not unique tracked fruits.")
        show_details(all_detections[:30])
        if len(all_detections) > 30:
            st.caption(f"Showing 30 of {len(all_detections)} sampled-frame detections.")
    finally:
        if capture is not None:
            capture.release()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


require_login()

with st.sidebar:
    st.success("Logged in")
    if st.button("Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; }
    .app-title {
        color: #32678d;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 750;
        text-align: center;
        margin-bottom: 0;
    }
    .app-subtitle {
        color: #667085;
        font-size: 1.15rem;
        text-align: center;
        margin: 0.4rem 0 2rem;
    }
    </style>
    <div class="app-title">🍎 Fruit Quality Inspector</div>
    <div class="app-subtitle">
        Upload an image or video to detect fruits and analyze available quality labels.
    </div>
    """,
    unsafe_allow_html=True,
)

if TRAINED_MODEL.is_file():
    model_path = TRAINED_MODEL
    use_pretrained_model = False
elif PRETRAINED_MODEL.is_file():
    model_path = PRETRAINED_MODEL
    use_pretrained_model = True
else:
    st.error(
        "No model weights were found. Add trained weights at "
        "'runs/detect/train/weights/best.pt' or restore 'yolov8n.pt'."
    )
    st.stop()

model = load_model(str(model_path))
input_type = st.radio("Input type", ["Image", "Video"], horizontal=True)
file_types = (
    ["jpg", "jpeg", "png"]
    if input_type == "Image"
    else ["mp4", "mov", "avi", "mpeg4"]
)
uploaded_file = st.file_uploader("Choose file", type=file_types)

if uploaded_file is None:
    st.info(f"No file uploaded yet. Please upload an {input_type.lower()} to analyze.")
elif input_type == "Image":
    show_image_result(uploaded_file, model, use_pretrained_model)
else:
    show_video_result(uploaded_file, model, use_pretrained_model)
