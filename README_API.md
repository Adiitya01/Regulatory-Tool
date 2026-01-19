# 🚀 FastAPI + React Setup Guide

This guide explains how to set up and run the new FastAPI backend with React frontend.

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+ and npm
- LM Studio running locally (default: `http://127.0.0.1:1234`)

## 🔧 Backend Setup (FastAPI)

### 1. Install Python Dependencies

```powershell
cd Backend
pip install -r requirements.txt
```

### 2. Run the FastAPI Server

```powershell
# Option 1: Using uvicorn directly
cd Backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python
python api.py
```

The API will be available at `http://localhost:8000`

### 3. API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎨 Frontend Setup (React)

### 1. Install Node Dependencies

```powershell
cd frontend
npm install
```

### 2. Run the Development Server

```powershell
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 3. Build for Production

```powershell
npm run build
```

## 📡 API Endpoints

### Health & Status
- `GET /api/health` - Health check
- `GET /api/llm/status` - Check LLM connection status
- `GET /api/files/status` - Get status of all output files

### Guideline Processing
- `POST /api/guideline/upload` - Upload and extract ISO 11135 guideline PDF
- `POST /api/guideline/polish` - Polish extracted guideline content using LLM

### DHF Processing
- `POST /api/dhf/upload` - Upload and extract DHF PDF

### Validation
- `POST /api/validation/run` - Run validation analysis

### File Management
- `GET /api/files/{filename}` - Download a generated file
- `GET /api/files/{filename}/content` - Get file content as JSON

## 🔄 Processing Flow

1. **Upload Guideline PDF** → Extract parameters
2. **Polish Guideline** → LLM enhancement
3. **Upload DHF PDF** → Extract DHF parameters
4. **Run Validation** → Generate compliance report

## 🛠️ Development

### Backend Development
- API code: `Backend/api.py`
- Uses existing backend modules: `Guideline_Extractor.py`, `LLM_Engine.py`, `DHF_Extractor.py`, `validation.py`
- CORS enabled for `localhost:3000` and `localhost:5173`

### Frontend Development
- Main app: `frontend/src/App.jsx`
- Styles: `frontend/src/App.css`
- Uses Vite for fast development
- Proxy configured for API calls

## 🐛 Troubleshooting

### Backend Issues
- **Port 8000 already in use**: Change port in `api.py` or `uvicorn` command
- **Import errors**: Ensure you're running from `Backend` directory or PYTHONPATH is set
- **File not found errors**: Check that `outputs/` directory exists

### Frontend Issues
- **API connection errors**: Ensure backend is running on port 8000
- **CORS errors**: Check CORS settings in `Backend/api.py`
- **Build errors**: Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`

### LM Studio Issues
- Ensure LM Studio is running
- Check model is loaded
- Verify API base URL matches (default: `http://127.0.0.1:1234`)

## 📝 Environment Variables

### Backend
```powershell
$env:LLM_API_BASE = "http://127.0.0.1:1234"
$env:LLM_MODEL_NAME = "meta-llama-3.1-8b-instruct"
```

### Frontend
No environment variables needed (uses proxy configuration)

## 🚀 Production Deployment

### Backend
```powershell
# Use production ASGI server
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend
```powershell
npm run build
# Serve the `dist` folder with a web server (nginx, Apache, etc.)
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)

