"""Local Streamlit interface for video violence screening."""

import tempfile
from pathlib import Path

import streamlit as st

from violence_detector import MODEL_ID, DetectorError, analyze_video, load_detector

st.set_page_config(page_title="Video Violence Screening", page_icon="🎞️")
st.title("Video Violence Screening")
st.write(
    "Choose a video to screen for possible fighting. Analysis runs locally "
    "using a pretrained RWF-2000 model."
)
st.warning(
    "This tool can make mistakes. Treat results as review flags, not proof of "
    "violence or a substitute for human judgment."
)


@st.cache_resource(show_spinner=False)
def get_detector():
    return load_detector()


uploaded = st.file_uploader(
    "Choose a video",
    type=("mp4", "mov", "avi", "mkv"),
    help="The video is processed locally and is not sent to an inference service.",
)
threshold = st.slider(
    "Flagging sensitivity",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="A window is flagged when its model confidence meets or exceeds this value.",
)

if uploaded is not None:
    st.caption(f"Selected: {uploaded.name}")
    if st.button("Analyze video", type="primary"):
        progress = st.progress(0, text="Loading the pretrained model…")
        try:
            detector = get_detector()
            progress.progress(10, text="Preparing video…")

            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix) as video_file:
                video_file.write(uploaded.getbuffer())
                video_file.flush()
                result = analyze_video(
                    video_file.name,
                    detector,
                    threshold=threshold,
                    progress_callback=lambda fraction: progress.progress(
                        10 + int(fraction * 90), text="Analyzing video…"
                    ),
                )

            progress.empty()
            if result["segments"]:
                st.error("Possible violence detected")
                st.write("Flagged time segments:")
                st.dataframe(
                    [
                        {
                            "Start": segment["start_label"],
                            "End": segment["end_label"],
                            "Confidence": f'{segment["confidence"]:.1%}',
                        }
                        for segment in result["segments"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.success("No likely violence detected above the selected threshold.")

            st.caption(
                f"Highest violence confidence: {result['max_confidence']:.1%}. "
                f"Model: {MODEL_ID}. Scores are estimates, not calibrated certainty."
            )
        except DetectorError as exc:
            progress.empty()
            st.error(str(exc))
        except Exception as exc:
            progress.empty()
            st.exception(exc)
