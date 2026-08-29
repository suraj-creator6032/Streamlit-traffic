import streamlit as st
import cv2
import tempfile
import os
from collections import Counter
from ultralytics import YOLO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="UrbanSense AI",
    page_icon="🚌",
    layout="wide"
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


# =========================================================
# VEHICLE CLASSES
# COCO CLASS IDs
# =========================================================

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# =========================================================
# VIDEO PROCESSING
# =========================================================

def process_video(input_video, output_video):

    model = load_model()

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise Exception("Unable to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    # Store unique vehicle IDs
    vehicle_ids = {
        "Bicycle": set(),
        "Car": set(),
        "Motorcycle": set(),
        "Bus": set(),
        "Truck": set()
    }

    frame_number = 0

    progress = st.progress(0)

    status = st.empty()

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # -------------------------------------------------
        # YOLO TRACKING
        # -------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False
        )

        result = results[0]

        # -------------------------------------------------
        # DETECTIONS
        # -------------------------------------------------

        if result.boxes is not None:

            boxes = result.boxes

            for i in range(len(boxes)):

                class_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                if class_id not in VEHICLE_CLASSES:
                    continue

                vehicle_name = VEHICLE_CLASSES[class_id]

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    boxes.xyxy[i].tolist()
                )

                # Tracking ID
                track_id = None

                if boxes.id is not None:
                    track_id = int(
                        boxes.id[i].item()
                    )

                    vehicle_ids[
                        vehicle_name
                    ].add(track_id)

                # -------------------------------------------------
                # DRAW BOX
                # -------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                if track_id is not None:

                    label = (
                        f"{vehicle_name} "
                        f"ID:{track_id} "
                        f"{confidence:.2f}"
                    )

                else:

                    label = (
                        f"{vehicle_name} "
                        f"{confidence:.2f}"
                    )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # -------------------------------------------------
        # TOTAL UNIQUE VEHICLES
        # -------------------------------------------------

        total = sum(
            len(ids)
            for ids in vehicle_ids.values()
        )

        # -------------------------------------------------
        # VIDEO OVERLAY
        # -------------------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (300, 65),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Vehicles Detected: {total}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        # -------------------------------------------------
        # PROGRESS
        # -------------------------------------------------

        if total_frames > 0:

            progress_value = (
                frame_number / total_frames
            )

            progress.progress(
                min(progress_value, 1.0)
            )

            status.text(
                f"Processing frame "
                f"{frame_number}/{total_frames}"
            )

    cap.release()
    writer.release()

    progress.empty()
    status.empty()

    return {
        "Total Vehicles": sum(
            len(ids)
            for ids in vehicle_ids.values()
        ),
        "Bicycles": len(
            vehicle_ids["Bicycle"]
        ),
        "Cars": len(
            vehicle_ids["Car"]
        ),
        "Motorcycles": len(
            vehicle_ids["Motorcycle"]
        ),
        "Buses": len(
            vehicle_ids["Bus"]
        ),
        "Trucks": len(
            vehicle_ids["Truck"]
        )
    }


# =========================================================
# HEADER
# =========================================================

st.title("🚌 UrbanSense AI")

st.subheader(
    "AI-Powered Mobile Urban Intelligence Platform"
)

st.write(
    "Transforming public transport buses into "
    "mobile urban sensing units."
)

st.divider()


# =========================================================
# VIDEO UPLOAD
# =========================================================

st.header("📹 Bus Camera Feed")

uploaded_file = st.file_uploader(
    "Upload road traffic footage captured by a bus-mounted camera",
    type=["mp4", "avi", "mov"]
)


# =========================================================
# SHOW INPUT VIDEO
# =========================================================

if uploaded_file is not None:

    st.video(uploaded_file)

    st.write("")

    analyze = st.button(
        "🚀 Analyze Traffic",
        type="primary",
        use_container_width=True
    )

    if analyze:

        # -------------------------------------------------
        # SAVE INPUT VIDEO
        # -------------------------------------------------

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_file.getbuffer()
        )

        input_file.close()

        input_path = input_file.name

        # -------------------------------------------------
        # OUTPUT VIDEO
        # -------------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_file.close()

        output_path = output_file.name

        # -------------------------------------------------
        # RUN AI
        # -------------------------------------------------

        with st.spinner(
            "Running YOLO vehicle detection..."
        ):

            try:

                counts = process_video(
                    input_path,
                    output_path
                )

                st.success(
                    "Traffic analysis completed!"
                )

                # =================================================
                # RESULTS
                # =================================================

                st.divider()

                st.header("📊 Traffic Intelligence")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric(
                    "Total Vehicles",
                    counts["Total Vehicles"]
                )

                col2.metric(
                    "Cars",
                    counts["Cars"]
                )

                col3.metric(
                    "Motorcycles",
                    counts["Motorcycles"]
                )

                col4.metric(
                    "Buses",
                    counts["Buses"]
                )

                col5, col6 = st.columns(2)

                col5.metric(
                    "Trucks",
                    counts["Trucks"]
                )

                col6.metric(
                    "Bicycles",
                    counts["Bicycles"]
                )

                # =================================================
                # OUTPUT VIDEO
                # =================================================

                st.divider()

                st.header("🤖 AI Detection Output")

                st.video(output_path)

                # =================================================
                # SIMPLE SUMMARY
                # =================================================

                st.divider()

                st.header("📋 Detection Summary")

                summary = {
                    "Vehicle Type": [
                        "Car",
                        "Motorcycle",
                        "Bus",
                        "Truck",
                        "Bicycle"
                    ],
                    "Count": [
                        counts["Cars"],
                        counts["Motorcycles"],
                        counts["Buses"],
                        counts["Trucks"],
                        counts["Bicycles"]
                    ]
                }

                st.table(summary)

            except Exception as e:

                st.error(
                    f"Processing error: {e}"
                )

            finally:

                if os.path.exists(input_path):
                    os.remove(input_path)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Prototype | Python • OpenCV • YOLO • Streamlit"
)