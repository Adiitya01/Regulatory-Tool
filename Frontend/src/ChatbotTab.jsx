import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');
const API_BASE_URL = (BASE_URL.endsWith('/') ? BASE_URL.slice(0, -1) : BASE_URL) + '/api';

console.log('Chatbot using API Base URL:', API_BASE_URL);

function ChatbotTab({ filesStatus }) {
    const [messages, setMessages] = useState([])
    const [inputMessage, setInputMessage] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [ragIngested, setRagIngested] = useState(false)
    const [isIngesting, setIsIngesting] = useState(false)
    const messagesEndRef = useRef(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const polishedExists = filesStatus['polished_regulatory_guidance.txt']?.exists
    const validationExists = filesStatus['validation_report.txt']?.exists

    // Only show "No Data" if we have received at least one status update from the backend
    // and both files are definitely missing.
    const hasReceivedStatus = Object.keys(filesStatus).length > 0
    const dataAvailable = polishedExists || validationExists

    const handleIngestData = async () => {
        setIsIngesting(true)
        try {
            const response = await axios.post(`${API_BASE_URL}/rag/ingest`)
            if (response.status === 200) {
                setRagIngested(true)
                addBotMessage("✅ Knowledge base loaded! I'm ready to answer your questions about ISO 11135 compliance.")
            }
        } catch (error) {
            console.error('Ingest error:', error);
            addBotMessage("❌ Failed to load knowledge base. Please ensure the backend is running and the necessary files exist.")
        } finally {
            setIsIngesting(false)
        }
    }

    const addBotMessage = (text) => {
        setMessages(prev => [...prev, { role: 'assistant', content: text, timestamp: new Date() }])
    }

    const addUserMessage = (text) => {
        setMessages(prev => [...prev, { role: 'user', content: text, timestamp: new Date() }])
    }

    const handleSendMessage = async (messageText = null) => {
        const textToSend = messageText || inputMessage
        if (!textToSend.trim()) return

        addUserMessage(textToSend)
        setInputMessage('')
        setIsLoading(true)

        try {
            const response = await axios.post(`${API_BASE_URL}/chat`, {
                message: textToSend
            })
            addBotMessage(response.data.response)
        } catch (error) {
            console.error('Chat error:', error);
            addBotMessage("❌ Sorry, I encountered an error processing your question. Please try again.")
        } finally {
            setIsLoading(false)
        }
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSendMessage()
        }
    }

    const exampleQuestions = [
        "Why is bioburden marked as missing?",
        "What is clause 9.4 in ISO 11135?",
        "Which documents do I need to add for IQ/OQ/PQ?",
        "Explain the validation requirements"
    ]

    if (hasReceivedStatus && !dataAvailable) {
        return (
            <div className="tab-panel">
                <h2>🤖 Regulatory Consultant Chatbot</h2>
                <div className="warning-box" style={{
                    marginTop: '2rem',
                    padding: '2rem',
                    background: '#fff9fa',
                    border: '1px solid #ffccd3',
                    borderRadius: '12px',
                    textAlign: 'center'
                }}>
                    <h3 style={{ color: '#d32f2f' }}>⚠️ No Data Available</h3>
                    <p>The chatbot needs processed data to provide answers. Please complete the pipeline first:</p>
                    <div style={{ maxWidth: '400px', margin: '1.5rem auto', textAlign: 'left' }}>
                        <ol style={{ lineHeight: '2' }}>
                            <li style={{ color: filesStatus['guideline_extraction_output.txt']?.exists ? '#2e7d32' : '#666' }}>
                                {filesStatus['guideline_extraction_output.txt']?.exists ? '✅' : '🔴'} Extract Guideline PDF
                            </li>
                            <li style={{ color: polishedExists ? '#2e7d32' : '#666' }}>
                                {polishedExists ? '✅' : '🔴'} Run LLM Polishing
                            </li>
                            <li style={{ color: filesStatus['DHF_Single_Extraction.txt']?.exists ? '#2e7d32' : '#666' }}>
                                {filesStatus['DHF_Single_Extraction.txt']?.exists ? '✅' : '🔴'} Extract DHF PDF
                            </li>
                            <li style={{ color: validationExists ? '#2e7d32' : '#666' }}>
                                {validationExists ? '✅' : '🔴'} Generate Validation Report
                            </li>
                        </ol>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="tab-panel chatbot-container">
            <h2>🤖 Regulatory Consultant Chatbot</h2>
            <p style={{ color: '#666', marginBottom: '1.5rem' }}>
                Ask questions about your compliance status, missing requirements, or ISO 11135 clauses.
            </p>

            {!ragIngested ? (
                <div className="chatbot-setup" style={{
                    padding: '2rem',
                    background: '#f8f9fa',
                    borderRadius: '12px',
                    border: '1px dashed #dee2e6',
                    textAlign: 'center'
                }}>
                    <div className="info-box">
                        <h3>💡 Ready to Consult</h3>
                        <p>I can analyze your reports and standards to provide expert guidance.</p>
                        <button
                            className="btn btn-primary"
                            onClick={handleIngestData}
                            disabled={isIngesting || !dataAvailable}
                            style={{ marginTop: '1rem', padding: '0.8rem 2rem' }}
                        >
                            {isIngesting ? '🔍 Loading Knowledge Base...' : '🔄 Load Knowledge Base'}
                        </button>
                        {!dataAvailable && <p style={{ fontSize: '0.8rem', color: '#d32f2f', marginTop: '0.5rem' }}>Wait for processing to complete</p>}
                    </div>
                </div>
            ) : (
                <>
                    <div className="chat-messages" style={{ minHeight: '400px', maxHeight: '500px', overflowY: 'auto', padding: '1rem', background: '#fff', borderRadius: '8px', border: '1px solid #eee' }}>
                        {messages.length === 0 && (
                            <div className="welcome-message" style={{ textAlign: 'center', padding: '3rem 1rem', color: '#888' }}>
                                <h3>👋 How can I help you today?</h3>
                                <p>You can ask about failures found in the report or specific ISO clauses.</p>
                            </div>
                        )}

                        {messages.map((msg, idx) => (
                            <div key={idx} className={`chat-message ${msg.role}`}>
                                <div className="message-content">
                                    {msg.role === 'user' ? (
                                        <><strong>You:</strong> {msg.content}</>
                                    ) : (
                                        <><strong>🤖 Consultant:</strong><br />{msg.content}</>
                                    )}
                                </div>
                            </div>
                        ))}

                        {isLoading && (
                            <div className="chat-message assistant">
                                <div className="message-content">
                                    <strong>🤖 Consultant:</strong><br />
                                    <span className="typing-indicator">Thinking...</span>
                                </div>
                            </div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>

                    <div className="chat-input-area">
                        <div className="input-wrapper">
                            <input
                                type="text"
                                value={inputMessage}
                                onChange={(e) => setInputMessage(e.target.value)}
                                onKeyPress={handleKeyPress}
                                placeholder="Ask a question..."
                                className="chat-input"
                                disabled={isLoading}
                            />
                            <button
                                className="btn btn-primary send-button"
                                onClick={() => handleSendMessage()}
                                disabled={isLoading || !inputMessage.trim()}
                            >
                                📤 Send
                            </button>
                            <button
                                className="btn btn-secondary"
                                onClick={() => setMessages([])}
                                disabled={messages.length === 0}
                            >
                                🗑️ Clear
                            </button>
                        </div>
                    </div>

                    <div className="example-questions">
                        <h4>💡 Example Questions:</h4>
                        <div className="examples-grid">
                            {exampleQuestions.map((question, idx) => (
                                <button
                                    key={idx}
                                    className="example-button"
                                    onClick={() => handleSendMessage(question)}
                                    disabled={isLoading}
                                >
                                    📝 {question}
                                </button>
                            ))}
                        </div>
                    </div>
                </>
            )}
        </div>
    )
}

export default ChatbotTab
