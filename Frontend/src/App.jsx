import React, { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import ChatbotTab from './ChatbotTab'

const BASE_URL = import.meta.env.VITE_API_URL || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');
const API_BASE_URL = (BASE_URL.endsWith('/') ? BASE_URL.slice(0, -1) : BASE_URL) + '/api';

console.log('Using API Base URL:', API_BASE_URL);

function App() {
  const [mainTab, setMainTab] = useState('processing')
  const [processingTab, setProcessingTab] = useState('guideline')
  const [selectedGuideline, setSelectedGuideline] = useState('iso11135')
  const [guidelines, setGuidelines] = useState([])
  const [llmStatus, setLlmStatus] = useState({ connected: false, message: 'Checking...' })
  const [filesStatus, setFilesStatus] = useState({})
  const [pipelineCompletion, setPipelineCompletion] = useState(null)
  const [loading, setLoading] = useState({})
  const [messages, setMessages] = useState([])
  const [selectedFile, setSelectedFile] = useState(null)
  const [fileContent, setFileContent] = useState('')

  useEffect(() => {
    loadGuidelines()
    checkLlmStatus()
    checkFilesStatus()
    checkPipelineCompletion()
    const interval = setInterval(() => {
      checkFilesStatus()
      checkPipelineCompletion()
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadGuidelines = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/guidelines`)
      setGuidelines(response.data.guidelines || [])
    } catch (error) {
      console.error('Error loading guidelines:', error)
    }
  }

  const checkLlmStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/llm/status`)
      setLlmStatus(response.data)
    } catch (error) {
      setLlmStatus({ connected: false, message: 'Error checking LLM status' })
    }
  }

  const checkFilesStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/files/status`)
      setFilesStatus(response.data.files)
    } catch (error) {
      console.error('Error checking files status:', error)
    }
  }

  const checkPipelineCompletion = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/pipeline/completion`)
      setPipelineCompletion(response.data)
    } catch (error) {
      console.error('Error checking pipeline completion:', error)
    }
  }

  const addMessage = (type, text) => {
    setMessages(prev => [...prev, { type, text, timestamp: new Date() }])
    setTimeout(() => {
      setMessages(prev => prev.slice(1))
    }, 5000)
  }

  const handleFileUpload = async (endpoint, file, fileType) => {
    if (!file) {
      addMessage('error', `Please select a ${fileType} file`)
      return
    }

    setLoading(prev => ({ ...prev, [endpoint]: true }))
    addMessage('info', `Uploading ${fileType}...`)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(`${API_BASE_URL}/${endpoint}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      addMessage('success', response.data.message)
      checkFilesStatus()
      checkPipelineCompletion()
    } catch (error) {
      addMessage('error', error.response?.data?.detail || `Error uploading ${fileType}`)
    } finally {
      setLoading(prev => ({ ...prev, [endpoint]: false }))
    }
  }

  const handleProcess = async (endpoint, processName) => {
    setLoading(prev => ({ ...prev, [endpoint]: true }))
    addMessage('info', `Starting ${processName}...`)

    try {
      const response = await axios.post(`${API_BASE_URL}/${endpoint}`)
      addMessage('success', response.data.message)
      checkFilesStatus()
      checkPipelineCompletion()
    } catch (error) {
      addMessage('error', error.response?.data?.detail || `Error during ${processName}`)
    } finally {
      setLoading(prev => ({ ...prev, [endpoint]: false }))
    }
  }

  const handleRunPipeline = async (guidelineFile, dhfFile) => {
    if (!guidelineFile || !dhfFile) {
      addMessage('error', 'Please select both guideline and DHF PDF files')
      return
    }

    setLoading(prev => ({ ...prev, 'pipeline/run': true }))
    addMessage('info', 'Starting full pipeline...')

    try {
      const formData = new FormData()
      formData.append('guideline_file', guidelineFile)
      formData.append('dhf_file', dhfFile)

      const response = await axios.post(`${API_BASE_URL}/pipeline/run`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 600000 // 10 minutes timeout for long-running pipeline
      })

      if (response.data.success) {
        addMessage('success', response.data.message)
        // Show step-by-step results
        response.data.steps.forEach((step, idx) => {
          if (step.status === 'completed') {
            setTimeout(() => {
              addMessage('success', `${step.name}: ${step.message}`)
            }, idx * 500)
          } else if (step.status === 'error') {
            addMessage('error', `${step.name}: ${step.error || step.message}`)
          }
        })
      } else {
        addMessage('error', response.data.message)
      }

      checkFilesStatus()
      checkPipelineCompletion()
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || 'Error running pipeline'
      addMessage('error', errorMsg)

      // If response has step details, show them
      if (error.response?.data?.steps) {
        error.response.data.steps.forEach(step => {
          if (step.status === 'error') {
            addMessage('error', `${step.name}: ${step.error || step.message}`)
          }
        })
      }
    } finally {
      setLoading(prev => ({ ...prev, 'pipeline/run': false }))
    }
  }

  const viewFile = async (filename) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/files/${filename}/content`)
      setFileContent(response.data.content)
      setSelectedFile(filename)
    } catch (error) {
      addMessage('error', `Error loading file: ${error.response?.data?.detail || error.message}`)
    }
  }

  const downloadFile = (filename) => {
    window.open(`${API_BASE_URL}/files/${filename}`, '_blank')
  }

  const getFileStatus = (filename) => {
    return filesStatus[filename]?.exists || false
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>DHF Document Validator</h1>
        <div className="header-logo">
          <img src="/ethosh_logo.png" alt="Ethosh Logo" className="brand-logo" />
        </div>
      </header>

      {messages.length > 0 && (
        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message message-${msg.type}`}>
              {msg.text}
            </div>
          ))}
        </div>
      )}

      <div className="app-container">
        <nav className="sidebar">
          <div className="status-section">
            <div className={`status-badge-sidebar ${llmStatus.connected ? 'online' : 'offline'}`}>
              <span className="dot"></span>
              {llmStatus.connected ? 'Systems Operational' : 'Offline'}
            </div>
          </div>

          <div className="nav-section">
            <h3>Select Guideline</h3>
            <select
              value={selectedGuideline}
              onChange={(e) => setSelectedGuideline(e.target.value)}
              className="guideline-select-sidebar"
            >
              <option value="">-- Choose Guideline --</option>
              {guidelines.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.name}
                </option>
              ))}
            </select>
          </div>

          <div className="nav-section">
            <h3>Control Panel</h3>
            <button
              className={`nav-button ${mainTab === 'processing' ? 'active' : ''}`}
              onClick={() => setMainTab('processing')}
            >
              Processing
            </button>
            <button
              className={`nav-button ${mainTab === 'files' ? 'active' : ''}`}
              onClick={() => setMainTab('files')}
            >
              Files
            </button>
            <button
              className={`nav-button ${mainTab === 'chatbot' ? 'active' : ''}`}
              onClick={() => setMainTab('chatbot')}
            >
              Chatbot
            </button>
          </div>

          <div className="nav-section">
            <h3>Pipeline Status</h3>
            {pipelineCompletion && (
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0.5rem' }}>
                  Overall Progress: {pipelineCompletion.overall_progress}%
                </div>
                <div style={{
                  width: '100%',
                  height: '8px',
                  backgroundColor: '#e0e0e0',
                  borderRadius: '4px',
                  overflow: 'hidden'
                }}>
                  <div style={{
                    width: `${pipelineCompletion.overall_progress}%`,
                    height: '100%',
                    backgroundColor: '#4caf50',
                    transition: 'width 0.3s ease'
                  }}></div>
                </div>
              </div>
            )}
            <div className="pipeline-status">
              {pipelineCompletion ? (
                pipelineCompletion.steps.map((step) => (
                  <div
                    key={step.step}
                    className={`status-item ${step.status === 'completed' ? 'complete' : ''}`}
                    title={step.message}
                  >
                    <span>{step.name.split(' ')[0]}</span>
                    <span>{step.status === 'completed' ? 'Complete' : 'Pending'}</span>
                  </div>
                ))
              ) : (
                <>
                  <div className={`status-item ${getFileStatus('guideline_extraction_output.txt') ? 'complete' : ''}`}>
                    <span>Guideline</span>
                    <span>{getFileStatus('guideline_extraction_output.txt') ? 'Complete' : 'Pending'}</span>
                  </div>
                  <div className={`status-item ${getFileStatus('polished_regulatory_guidance.txt') ? 'complete' : ''}`}>
                    <span>Polishing</span>
                    <span>{getFileStatus('polished_regulatory_guidance.txt') ? 'Complete' : 'Pending'}</span>
                  </div>
                  <div className={`status-item ${getFileStatus('DHF_Single_Extraction.txt') ? 'complete' : ''}`}>
                    <span>DHF</span>
                    <span>{getFileStatus('DHF_Single_Extraction.txt') ? 'Complete' : 'Pending'}</span>
                  </div>
                  <div className={`status-item ${getFileStatus('validation_report.txt') ? 'complete' : ''}`}>
                    <span>Validation</span>
                    <span>{getFileStatus('validation_report.txt') ? 'Complete' : 'Pending'}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </nav>

        <main className="main-content">
          {mainTab === 'processing' && (
            <div className="processing-tabs">
              <div className="tab-buttons">
                <button
                  className={`tab-button ${processingTab === 'pipeline' ? 'active' : ''}`}
                  onClick={() => setProcessingTab('pipeline')}
                >
                  Full Pipeline
                </button>
                <button
                  className={`tab-button ${processingTab === 'guideline' ? 'active' : ''}`}
                  onClick={() => setProcessingTab('guideline')}
                >
                  Guideline
                </button>
                <button
                  className={`tab-button ${processingTab === 'polishing' ? 'active' : ''}`}
                  onClick={() => setProcessingTab('polishing')}
                >
                  Polishing
                </button>
                <button
                  className={`tab-button ${processingTab === 'dhf' ? 'active' : ''}`}
                  onClick={() => setProcessingTab('dhf')}
                >
                  DHF
                </button>
                <button
                  className={`tab-button ${processingTab === 'validation' ? 'active' : ''}`}
                  onClick={() => setProcessingTab('validation')}
                >
                  Validation
                </button>
              </div>

              <div className="tab-content">
                {processingTab === 'pipeline' && (
                  <PipelineTab
                    onRunPipeline={handleRunPipeline}
                    loading={loading}
                    filesStatus={filesStatus}
                    onView={viewFile}
                    onDownload={downloadFile}
                  />
                )}
                {processingTab === 'guideline' && (
                  <GuidelineTab
                    onUpload={handleFileUpload}
                    onProcess={handleProcess}
                    loading={loading}
                    fileExists={getFileStatus('guideline_extraction_output.txt')}
                    onView={viewFile}
                    onDownload={downloadFile}
                  />
                )}
                {processingTab === 'polishing' && (
                  <PolishingTab
                    onProcess={handleProcess}
                    loading={loading}
                    fileExists={getFileStatus('polished_regulatory_guidance.txt')}
                    prerequisiteExists={getFileStatus('guideline_extraction_output.txt')}
                    onView={viewFile}
                    onDownload={downloadFile}
                  />
                )}
                {processingTab === 'dhf' && (
                  <DHFTab
                    onUpload={handleFileUpload}
                    loading={loading}
                    fileExists={getFileStatus('DHF_Single_Extraction.txt')}
                    onView={viewFile}
                    onDownload={downloadFile}
                  />
                )}
                {processingTab === 'validation' && (
                  <ValidationTab
                    onProcess={handleProcess}
                    loading={loading}
                    fileExists={getFileStatus('validation_report.txt')}
                    prerequisitesExist={getFileStatus('polished_regulatory_guidance.txt') && getFileStatus('DHF_Single_Extraction.txt')}
                    onView={viewFile}
                    onDownload={downloadFile}
                  />
                )}
              </div>
            </div>
          )}

          {mainTab === 'files' && (
            <FilesTab
              filesStatus={filesStatus}
              onView={viewFile}
              onDownload={downloadFile}
            />
          )}

          {mainTab === 'chatbot' && (
            <ChatbotTab
              filesStatus={filesStatus}
            />
          )}
        </main>
      </div>

      {selectedFile && (
        <FileViewerModal
          filename={selectedFile}
          content={fileContent}
          onClose={() => {
            setSelectedFile(null)
            setFileContent('')
          }}
          onDownload={downloadFile}
        />
      )}
    </div>
  )
}

function PipelineTab({ onRunPipeline, loading, filesStatus, onView, onDownload }) {
  const [guidelineFile, setGuidelineFile] = useState(null)
  const [dhfFile, setDhfFile] = useState(null)

  const getFileStatus = (filename) => {
    return filesStatus[filename]?.exists || false
  }

  return (
    <div className="tab-panel">
      <h2>Full Pipeline Processing</h2>

      <div className="pipeline-upload-section">
        <div className="upload-group">
          <label>Guideline PDF</label>
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setGuidelineFile(e.target.files[0])}
            className="file-input"
          />
          {guidelineFile && (
            <div className="file-info">Selected: {guidelineFile.name}</div>
          )}
        </div>

        <div className="upload-group">
          <label>DHF PDF</label>
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setDhfFile(e.target.files[0])}
            className="file-input"
          />
          {dhfFile && (
            <div className="file-info">Selected: {dhfFile.name}</div>
          )}
        </div>
      </div>

      <div className="button-group" style={{ marginTop: '2rem' }}>
        <button
          className="btn btn-primary btn-large"
          onClick={() => onRunPipeline(guidelineFile, dhfFile)}
          disabled={!guidelineFile || !dhfFile || loading['pipeline/run']}
        >
          {loading['pipeline/run'] ? 'Running Pipeline...' : 'Run Full Pipeline'}
        </button>
      </div>

      {loading['pipeline/run'] && (
        <div className="pipeline-progress">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: '0%' }}></div>
          </div>
          <p>Processing... This may take several minutes.</p>
        </div>
      )}

      <div className="pipeline-results" style={{ marginTop: '2rem' }}>
        <h3>Generated Files</h3>
        <div className="results-grid">
          <div className={`result-card ${getFileStatus('guideline_extraction_output.txt') ? 'complete' : ''}`}>
            <div className="result-name">Guideline Extraction</div>
            <div className="result-status">
              {getFileStatus('guideline_extraction_output.txt') ? 'Complete' : 'Pending'}
            </div>
            {getFileStatus('guideline_extraction_output.txt') && (
              <div className="result-actions">
                <button className="btn-small" onClick={() => onView('guideline_extraction_output.txt')}>View</button>
                <button className="btn-small" onClick={() => onDownload('guideline_extraction_output.txt')}>Download</button>
              </div>
            )}
          </div>

          <div className={`result-card ${getFileStatus('polished_regulatory_guidance.txt') ? 'complete' : ''}`}>
            <div className="result-name">Polished Guidance</div>
            <div className="result-status">
              {getFileStatus('polished_regulatory_guidance.txt') ? 'Complete' : 'Pending'}
            </div>
            {getFileStatus('polished_regulatory_guidance.txt') && (
              <div className="result-actions">
                <button className="btn-small" onClick={() => onView('polished_regulatory_guidance.txt')}>View</button>
                <button className="btn-small" onClick={() => onDownload('polished_regulatory_guidance.txt')}>Download</button>
              </div>
            )}
          </div>

          <div className={`result-card ${getFileStatus('DHF_Single_Extraction.txt') ? 'complete' : ''}`}>
            <div className="result-name">DHF Extraction</div>
            <div className="result-status">
              {getFileStatus('DHF_Single_Extraction.txt') ? 'Complete' : 'Pending'}
            </div>
            {getFileStatus('DHF_Single_Extraction.txt') && (
              <div className="result-actions">
                <button className="btn-small" onClick={() => onView('DHF_Single_Extraction.txt')}>View</button>
                <button className="btn-small" onClick={() => onDownload('DHF_Single_Extraction.txt')}>Download</button>
              </div>
            )}
          </div>

          <div className={`result-card ${getFileStatus('validation_report.txt') ? 'complete' : ''}`}>
            <div className="result-name">Validation Report</div>
            <div className="result-status">
              {getFileStatus('validation_report.txt') ? 'Complete' : 'Pending'}
            </div>
            {getFileStatus('validation_report.txt') && (
              <div className="result-actions">
                <button className="btn-small" onClick={() => onView('validation_report.txt')}>View</button>
                <button className="btn-small" onClick={() => onDownload('validation_report.txt')}>Download</button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function GuidelineTab({ onUpload, loading, fileExists, onView, onDownload }) {
  const [file, setFile] = useState(null)

  return (
    <div className="tab-panel">
      <h2>ISO 11135 Guideline Extraction</h2>

      {fileExists ? (
        <div className="extraction-complete">
          <div className="success-message">
            <h3>Guideline Extraction Complete</h3>
            <p>The guideline parameters have been successfully extracted.</p>
          </div>
          <div className="button-group">
            <button className="btn btn-secondary" onClick={() => onView('guideline_extraction_output.txt')}>
              View Results
            </button>
            <button className="btn btn-secondary" onClick={() => onDownload('guideline_extraction_output.txt')}>
              Download
            </button>
          </div>
        </div>
      ) : (
        <div className="upload-section">
          <h3>Upload Guideline PDF</h3>
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
            className="file-input"
          />
          <div className="button-group">
            <button
              className="btn btn-primary"
              onClick={() => onUpload('guideline/upload', file, 'guideline')}
              disabled={!file || loading['guideline/upload']}
            >
              {loading['guideline/upload'] ? 'Processing...' : 'Extract Parameters'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function PolishingTab({ onProcess, loading, fileExists, prerequisiteExists, onView, onDownload }) {
  return (
    <div className="tab-panel">
      <h2>LLM-Powered Content Polishing</h2>
      {!prerequisiteExists && (
        <div className="warning">Complete Guideline Processing first</div>
      )}
      <div className="button-group">
        <button
          className="btn btn-primary"
          onClick={() => onProcess('guideline/polish', 'polishing')}
          disabled={!prerequisiteExists || loading['guideline/polish']}
        >
          {loading['guideline/polish'] ? 'Polishing...' : 'Polish Content'}
        </button>
        {fileExists && (
          <>
            <button className="btn btn-secondary" onClick={() => onView('polished_regulatory_guidance.txt')}>
              View
            </button>
            <button className="btn btn-secondary" onClick={() => onDownload('polished_regulatory_guidance.txt')}>
              Download
            </button>
          </>
        )}
      </div>
    </div>
  )
}

function DHFTab({ onUpload, loading, fileExists, onView, onDownload }) {
  const [file, setFile] = useState(null)

  return (
    <div className="tab-panel">
      <h2>DHF Document Extraction</h2>

      {fileExists ? (
        <div className="extraction-complete">
          <div className="success-message">
            <h3>DHF Extraction Complete</h3>
            <p>The DHF document has been successfully extracted.</p>
          </div>
          <div className="button-group">
            <button className="btn btn-secondary" onClick={() => onView('DHF_Single_Extraction.txt')}>
              View Results
            </button>
            <button className="btn btn-secondary" onClick={() => onDownload('DHF_Single_Extraction.txt')}>
              Download
            </button>
          </div>
        </div>
      ) : (
        <div className="upload-section">
          <h3>Upload DHF PDF</h3>
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
            className="file-input"
          />
          <div className="button-group">
            <button
              className="btn btn-primary"
              onClick={() => onUpload('dhf/upload', file, 'DHF')}
              disabled={!file || loading['dhf/upload']}
            >
              {loading['dhf/upload'] ? 'Processing...' : 'Extract DHF'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function ValidationTab({ onProcess, loading, fileExists, prerequisitesExist, onView, onDownload }) {
  return (
    <div className="tab-panel">
      <h2>Generate Compliance Report</h2>
      {!prerequisitesExist && (
        <div className="warning">Need Polished Guidelines and DHF Extraction</div>
      )}
      <div className="button-group">
        <button
          className="btn btn-primary"
          onClick={() => onProcess('validation/run', 'validation')}
          disabled={!prerequisitesExist || loading['validation/run']}
        >
          {loading['validation/run'] ? 'Validating...' : 'Generate Report'}
        </button>
        {fileExists && (
          <>
            <button className="btn btn-secondary" onClick={() => onView('validation_report.txt')}>
              View Report
            </button>
            <button className="btn btn-secondary" onClick={() => onDownload('validation_report.txt')}>
              Download
            </button>
          </>
        )}
      </div>
    </div>
  )
}

function FilesTab({ filesStatus, onView, onDownload }) {
  const files = [
    { name: 'guideline_extraction_output.txt', label: 'Guideline Extraction' },
    { name: 'polished_regulatory_guidance.txt', label: 'Polished Guidance' },
    { name: 'DHF_Single_Extraction.txt', label: 'DHF Extraction' },
    { name: 'validation_report.txt', label: 'Validation Report' },
    { name: 'validation_terminal_output.txt', label: 'Terminal Output' }
  ]

  return (
    <div className="tab-panel">
      <h2>Generated Files</h2>
      <table className="files-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Status</th>
            <th>Size</th>
            <th>Modified</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {files.map(file => {
            const fileInfo = filesStatus[file.name]
            return (
              <tr key={file.name}>
                <td>{file.label}</td>
                <td>
                  <span className={`status-badge ${fileInfo?.exists ? 'complete' : 'pending'}`}>
                    {fileInfo?.exists ? 'Complete' : 'Pending'}
                  </span>
                </td>
                <td>{fileInfo?.size ? `${(fileInfo.size / 1024).toFixed(1)} KB` : '-'}</td>
                <td>{fileInfo?.modified ? new Date(fileInfo.modified).toLocaleString() : '-'}</td>
                <td>
                  {fileInfo?.exists && (
                    <>
                      <button className="btn-small" onClick={() => onView(file.name)}>View</button>
                      <button className="btn-small" onClick={() => onDownload(file.name)}>Download</button>
                    </>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function FileViewerModal({ filename, content, onClose, onDownload }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{filename}</h3>
          <div className="modal-actions">
            <button className="btn btn-secondary" onClick={() => onDownload(filename)}>
              Download
            </button>
            <button className="btn btn-secondary" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
        <div className="modal-body">
          <pre className="file-content">{content}</pre>
        </div>
      </div>
    </div>
  )
}

export default App
