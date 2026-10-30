# AI Media Forensics & Deepfake Detection (Starter MVP)

A database-free starter project with:
- React + Vite frontend
- FastAPI backend
- Open-source Hugging Face image-classification model
- Image analysis
- Video analysis by sampling frames
- Optional face-crop analysis using OpenCV Haar cascades
- Downloadable JSON report

## Important limitations
This is a research/demo tool, not a reliable authenticity oracle. A classifier's "real/fake" score is not proof. It may fail on unseen generators, edited backgrounds, compression, screenshots, or face swaps. Face-crop analysis only checks whether the selected model finds the crop suspicious; it does not prove that a face was replaced. Background-only detection is not implemented as a validated forensic model in this MVP.

## Requirements
- Python 3.10 or 3.11 recommended
- Node.js 18+ recommended
- Internet for the first model download
- CPU works but can be slow; GPU is optional

## 1. Start the backend

Windows PowerShell:
```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

macOS/Linux:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/health to check the backend.

### Model
By default, the backend attempts to load `umm-maybe/AI-image-detector` from Hugging Face. The model downloads on first use. You can override it with the `MODEL_ID` environment variable if needed.

Windows PowerShell:
```powershell
$env:MODEL_ID="umm-maybe/AI-image-detector"
```

macOS/Linux:
```bash
export MODEL_ID="umm-maybe/AI-image-detector"
```

If the model is unavailable or incompatible, check the model card and choose a compatible image-classification model whose license permits your intended use.

## 2. Start the frontend

Open another terminal:
```bash
cd frontend
npm install
npm run dev
```
Open the local URL printed by Vite, usually http://localhost:5173.

The frontend calls the backend at `http://127.0.0.1:8000`. For a deployed backend, set `VITE_API_URL` before building:
```bash
VITE_API_URL=https://your-backend.example.com npm run build
```
In Windows PowerShell:
```powershell
$env:VITE_API_URL="https://your-backend.example.com"
npm run build
```

## API
- `GET /health`
- `POST /api/analyze/image` — multipart field `file`
- `POST /api/analyze/video` — multipart field `file`, optional form field `max_frames` (default 12, maximum 40)

## Privacy
Files are processed in memory or temporary files and are deleted after analysis. Reports are returned to the browser; this app does not save a history or use a database. Avoid uploading sensitive media to any public deployment unless you understand its hosting and privacy policies.

## Free deployment notes
Free hosting tiers have changing quotas and may sleep, run out of RAM, or be too slow for model inference. Start locally first. Do not expose a public upload endpoint without adding file-size limits, rate limiting, authentication if needed, and stronger security controls.
