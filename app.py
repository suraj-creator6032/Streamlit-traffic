import streamlit as st
import tempfile
import os
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
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


# =========================================================
# VEHICLE CLASSES
# =========================================================

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
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
# PROCESS
# =========================================================

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button(
        "🚀 Analyze Traffic",
        type="primary",
        use_container_width=True
    ):

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(
            uploaded_file.getbuffer()
        )

        input_file.close()

        input_path = input_file.name

        try:

            with st.spinner(
                "Loading AI model..."
            ):

                model = load_model()

            st.info(
                "AI is detecting and tracking vehicles..."
            )

            # -------------------------------------------------
            # RUN YOLO
            # -------------------------------------------------

            results = model.track(
                source=input_path,
                classes=list(VEHICLE_CLASSES.keys()),
                tracker="bytetrack.yaml",
                persist=True,
                save=True,
                project="runs",
                name="traffic",
                exist_ok=True,
                verbose=False,
                stream=True
            )

            # -------------------------------------------------
            # COUNT DETECTED VEHICLES
            # -------------------------------------------------

            vehicle_ids = {
                "Bicycle": set(),
                "Car": set(),
                "Motorcycle": set(),
                "Bus": set(),
                "Truck": set()
            }

            for result in results:

                if result.boxes is None:
                    continue

                boxes = result.boxes

                for i in range(len(boxes)):

                    class_id = int(
                        boxes.cls[i].item()
                    )

                    if class_id not in VEHICLE_CLASSES:
                        continue

                    vehicle_name = VEHICLE_CLASSES[class_id]

                    if boxes.id is not None:

                        track_id = int(
                            boxes.id[i].item()
                        )

                        vehicle_ids[
                            vehicle_name
                        ].add(track_id)

            # -------------------------------------------------
            # COUNTS
            # -------------------------------------------------

            counts = {
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

            counts["Total Vehicles"] = sum(
                counts.values()
            )

            # -------------------------------------------------
            # RESULTS
            # -------------------------------------------------

            st.success(
                "Traffic analysis completed!"
            )

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

            # -------------------------------------------------
            # OUTPUT VIDEO
            # -------------------------------------------------

            output_directory = (
                "runs/traffic"
            )

            possible_files = []

            if os.path.exists(output_directory):

                for filename in os.listdir(
                    output_directory
                ):

                    if filename.endswith(
                        (".mp4", ".avi", ".mov")
                    ):

                        possible_files.append(
                            os.path.join(
                                output_directory,
                                filename
                            )
                        )

            if possible_files:

                output_video = possible_files[0]

                st.divider()

                st.header(
                    "🤖 AI Detection Output"
                )

                st.video(output_video)

            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------

            st.divider()

            st.header("📋 Detection Summary")

            st.table({
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
            })

        except Exception as e:

            st.error(
                "An error occurred while processing "
                "the video."
            )

            st.exception(e)

        finally:

            if os.path.exists(input_path):
                os.remove(input_path)


else:

    st.info(
        "Upload a road traffic video to begin."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "SIH Prototype | Python • YOLO • Streamlit"
)