# Local Video Violence Screening

A small Streamlit app that screens a video locally and reports possible violence in short time segments with confidence scores. It uses OpenCV for video decoding and a pretrained 3D ResNet-18 checkpoint fine-tuned on RWF-2000.

> This is a review aid, not proof of violence or a substitute for human judgment. The checkpoint's reported evaluation results are self-reported and may not generalize to every kind of video.

## Run locally

Python 3.10 or later is recommended. From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run main.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit. On first analysis, the app downloads the model checkpoint from Hugging Face (about 400 MB); subsequent runs reuse the cached file. Inference uses CUDA when available and otherwise runs on CPU.

The app samples 16 frames from overlapping 2.5-second windows, advancing by 1 second. Windows at or above the selected confidence threshold (default 50%) are merged into flagged time segments. Confidence is a model estimate, not a calibrated probability of a real-world event.

## Model and dataset notes

- Checkpoint: [`Abdullahtahir11/rwf2000-violence-detection-3dresnet`](https://huggingface.co/Abdullahtahir11/rwf2000-violence-detection-3dresnet), MIT-licensed model card/checkpoint, using an RWF-2000-fine-tuned 3D ResNet-18.
- RWF-2000 raw videos are not publicly distributed from the dataset's project page; the dataset maintainers require approval for access and restrict redistribution/commercial use. This app does not download raw dataset videos.
- The model card reports 90% accuracy and 0.9649 ROC-AUC on RWF-2000. These figures are self-reported, and the model may flag sports or other vigorous motion.

## Notes

- Videos are uploaded to the local Streamlit process and analyzed on your computer; this project does not send them to a hosted inference service.
- Supported file extensions in the interface: MP4, MOV, AVI, and MKV. Actual codec support depends on the local OpenCV build.
- The default upload size limit is Streamlit's default. For larger files, set `server.maxUploadSize` in Streamlit configuration.
