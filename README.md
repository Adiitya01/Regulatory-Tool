# Regulatory Compliance Verification Tool (ISO 11135)

## 🏥 Overview
This is an AI-powered regulatory compliance verification tool designed to help MedTech companies ensure their **Device History Files (DHF)** meet the strict standards of **ISO 11135** (Ethylene Oxide Sterilization).

The tool parses PDF documents, extracts critical process parameters, validates them against the official ISO guidelines, and provides a **RAG-based Chatbot** to answer compliance questions and identify gaps.

## 🚀 Key Features

*   **📄 PDF Extraction:** Automatically extracts text and parameters from complex regulatory PDFs and DHF documents.
*   **🤖 AI Polishing:** Uses LLMs (Large Language Models) to structure and polish extracted raw data into standard regulatory formats.
*   **✅ Automated Validation:** Performs a multi-layer gap analysis to check if the DHF meets specific ISO requirements (e.g., IQ/OQ/PQ, Biological Indicators).
*   **💬 RAG Chatbot (Consultant):** An embedded "Regulatory Consultant" that you can chat with. It knows the context of *your* uploaded documents and the ISO standard, offering specific advice on missing evidence.
*   **☁️ Cloud-Ready:** Supports deployment on Render with Supabase for persistent storage of reports and extracted data.

## 🛠️ Tech Stack

### Backend
*   **Framework:** FastAPI (Python)
*   **LLM Integration:** Hugging Face API / LM Studio (Local Inference)
*   **RAG:** ChromaDB + SentenceTransformers (Vector Search)
*   **PDF Processing:** PyMuPDF, pdfplumber
*   **Storage:** Local Filesystem / Supabase Storage (Cloud)

### Frontend
*   **Framework:** React (Vite) / Streamlit (Prototype)
*   **Styling:** TailwindCSS / Custom CSS

## 📦 Installation & Setup

### Prerequisites
*   Python 3.10+
*   Node.js & npm (for Frontend)
*   Supabase Account (optional, for cloud storage)

### 1. Clone the Repository
```bash
git clone https://github.com/Adiitya01/Regulatory-Tool.git
cd Regulatory-Tool
```

### 2. Backend Setup
Navigate to the backend directory and set up the virtual environment:
```bash
cd Backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in `Backend/` with your keys:
```env
# Optional: Hugging Face Token for LLM
HF_TOKEN=your_token_here

# Storage Provider (local or supabase)
STORAGE_PROVIDER=local
# SUPABASE_URL=...
# SUPABASE_KEY=...
```

Run the Server:
```bash
python -m uvicorn api:app --reload --port 8000
```
The API will be available at `http://localhost:8000`.

### 3. Frontend Setup
Navigate to the frontend directory:
```bash
cd ../Frontend
npm install
npm run dev
```
The UI typically runs at `http://localhost:5173`.

## 📝 Usage Workflow
1.  **Upload Guideline:** Upload the ISO 11135 PDF (or use the one provided).
2.  **Upload DHF:** Upload your Validation Report or DHF document.
3.  **Process:** Click "Run Validation" to extract and analyze data.
4.  **Review:** view the generated `Validation Report` with scores and missing items.
5.  **Chat:** Ask the AI Consultant specific questions like "What documents are missing for OQ?"

## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
