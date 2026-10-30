# MediaForensicsAI – AI Media Forensics & Deepfake Detection

**Look closer. Question what you see.**

MediaForensicsAI is an experimental AI-powered media analysis application built using React, FastAPI, computer vision, and ONNX Runtime. It analyzes uploaded images and videos for model-detected indicators associated with AI-generated content.

The application provides separate image-level classification, face detection and face-crop analysis, and sampled video-frame analysis to support media review.

> **Disclaimer:** MediaForensicsAI is an experimental research and demonstration project. Its predictions are estimates and should not be treated as definitive proof that media is authentic or manipulated.

---

## Features

- **AI-Generated Image Detection:** Classifies the complete image using a pretrained ONNX image-classification model.
- **Face Detection:** Detects visible face regions using OpenCV.
- **Face-Crop Analysis:** Performs separate fake-vs-real estimates on detected face crops.
- **Video Analysis:** Samples selected frames from uploaded videos and analyzes them individually.
- **Evidence Timeline:** Highlights sampled frames with elevated model signals for further review.
- **Temporal Face Tracking:** Uses heuristic association to summarize detected faces across sampled frames.
- **Combined Image Assessment:** Compares overall image predictions with available face-crop predictions to identify agreement or conflicting signals.
- **Analysis Metadata:** Displays file and processing information.
- **Report Export:** Download analysis results as JSON or use the browser's print dialog to save a PDF report.
- **No Database:** Analysis history is not stored by the application.

---

## Technology Stack

### Frontend
- React.js
- Vite
- JavaScript
- CSS

### Backend
- Python
- FastAPI
- OpenCV
- Pillow
- NumPy
- ONNX Runtime
- Hugging Face Hub

### Machine Learning Model
- **Model Repository:** [Deckard618/imdet](https://huggingface.co/Deckard618/imdet)
- **Model Variant:** `pe_tiny`
- **Inference Format:** ONNX
- **Inference:** CPU-supported

### Deployment
- **Frontend:** [Netlify](https://www.netlify.com/)
- **Backend:** [Render](https://render.com/)
- **Model Files:** [Hugging Face Hub](https://huggingface.co/)

---

## Project Architecture

The application follows a frontend-backend architecture.

1. The user uploads an image or video through the React frontend.
2. The frontend sends the uploaded file to the FastAPI backend.
3. The backend processes the media using OpenCV and the ONNX model.
4. For images, the backend performs overall image classification and face-crop analysis where applicable.
5. For videos, the backend samples selected frames and analyzes them individually.
6. The backend returns the analysis results as JSON.
7. The frontend displays predictions, confidence scores, frame information, and evidence details.

---

## Project Structure

```text
AI_Detector/
│
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── ...
│   ├── package.json
│   └── index.html
│
└── README.md
```

---

## Requirements

Before running the project, install the following:

- Python 3.10 or Python 3.11 recommended
- Node.js 18 or later
- npm
- Git
- Internet connection for the initial model download
- CPU-supported inference (GPU is not required)

---

## Installation and Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/NehanShaikh/AI_Detector.git

cd AI_Detector
```

### 2. Run the Backend Locally

Open a terminal and navigate to the backend directory.

**Windows PowerShell**

```powershell
cd backend

py -m venv .venv

.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt

uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

**macOS / Linux**

```bash
cd backend

python3 -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Once the backend starts, access:

- **Health Check:** http://127.0.0.1:8000/health
- **API Documentation:** http://127.0.0.1:8000/docs

### 3. Run the Frontend Locally

Open another terminal from the project directory.

```bash
cd frontend

npm install

npm run dev
```

Open the local URL printed by Vite, usually:

http://localhost:5173

### 4. Frontend API Configuration

The frontend uses the `VITE_API_URL` environment variable to communicate with the backend.

For local development, the default API URL is:

```text
http://127.0.0.1:8000
```

For a deployed backend, create a `.env` file inside the frontend directory:

```env
VITE_API_URL=https://ai-media-forensics-api.onrender.com
```

Restart the Vite development server after changing the environment variable.

---

## Model Information

MediaForensicsAI uses an ONNX-based model from the Hugging Face repository:

**Model:** [Deckard618/imdet](https://huggingface.co/Deckard618/imdet)

**Selected Variant:** `pe_tiny`

The selected model provides an AI-generated-vs-real classification estimate.

The model is downloaded from its public Hugging Face repository when required by the backend.

The model output is an estimate and should not be interpreted as a definitive authenticity result.

For more information about model architecture, usage, and licensing, refer to the original Hugging Face model repository.

---

## API Endpoints

The backend exposes the following API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks backend health |
| POST | `/api/analyze/image` | Analyzes an uploaded image |
| POST | `/api/analyze/video` | Analyzes sampled frames from an uploaded video |

### Image Analysis

**Endpoint:**

```text
POST /api/analyze/image
```

**Request:**
- Multipart form field: `file`
- File: Image to analyze

### Video Analysis

**Endpoint:**

```text
POST /api/analyze/video
```

**Request:**
- Multipart form field: `file`
- Optional form field: `max_frames`

The frontend allows users to select the number of sampled frames for video analysis.

---

## Deployment

MediaForensicsAI is deployed using Netlify for the frontend and Render for the backend.

### Frontend Deployment – Netlify

**Live Website:** https://mediaforensicsai.netlify.app/

Deployment steps:

1. Push the project to GitHub.
2. Import the repository into Netlify.
3. Configure the frontend build settings.

| Setting | Value |
|---|---|
| Base directory | `frontend` |
| Build command | `npm run build` |
| Publish directory | `dist` |

Add the following environment variable in Netlify:

```env
VITE_API_URL=https://ai-media-forensics-api.onrender.com
```

Save the environment variable and deploy the frontend.

### Backend Deployment – Render

Create a new Web Service connected to the GitHub repository.

Configure the service:

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Root directory | `backend` |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |

Add the following environment variable:

```env
CORS_ORIGINS=https://mediaforensicsai.netlify.app
```

Make sure the CORS origin matches the actual deployed Netlify domain.

After deployment, verify the backend using:

https://ai-media-forensics-api.onrender.com/health

> **Hosting Note:** Free hosting services may sleep after inactivity, have memory limits, or take time to initialize model inference. Initial requests may take longer while the backend starts or downloads the model.

---

## Important Limitations

MediaForensicsAI is a research and demonstration project, not a reliable authenticity oracle.

- Model scores are estimates, not proof that media is real or fake.
- Models may fail on unseen image generators, edited images, screenshots, compression, and other transformations.
- Face-crop analysis provides an AI-generated-vs-real estimate. It is not a validated face-swap or identity-replacement detector.
- Video analysis examines sampled frames, not every frame.
- Temporal face tracking is heuristic and does not establish that a video has been manipulated.
- Background-only manipulation detection is not implemented as a validated forensic model.
- Different models may produce different or conflicting predictions.
- Agreement between image-level and face-crop predictions does not prove authenticity.

Results should be reviewed carefully and should not be used as the sole basis for decisions about authenticity.

---

## Privacy

- The application does not maintain a user account or analysis-history database.
- Analysis results are returned to the browser.
- Files may be processed in memory or through temporary files by the backend.
- Temporary files should be removed after processing.
- Public hosting providers may have their own infrastructure logs and operational data-retention policies.

Avoid uploading sensitive or private media to a public deployment unless you understand the applicable hosting and privacy policies.

---

## Future Improvements

- Improved deepfake and face-manipulation detection models.
- More robust video-level temporal analysis.
- Better detection of manipulated image regions.
- Additional model comparisons and evaluation.
- Performance and memory optimization.
- Enhanced reporting and forensic evidence visualization.

---

## Developer

**Nehan Shaikh**

Software Engineer (AI & ML)

**Project:** MediaForensicsAI

**GitHub:** https://github.com/NehanShaikh/AI_Detector

**Live Website:** https://mediaforensicsai.netlify.app/

---

## License and Credits

This project uses open-source software libraries and pretrained machine-learning models.

Refer to the respective repositories and model cards for their licenses and usage conditions.

Special credit to the developers and maintainers of the open-source technologies used in this project, including React, FastAPI, OpenCV, ONNX Runtime, and Hugging Face.

---

**MediaForensicsAI – Look closer. Question what you see.**
