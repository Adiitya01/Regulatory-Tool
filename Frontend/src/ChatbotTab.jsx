import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');
const API_BASE_URL = (BASE_URL.endsWith('/') ? BASE_URL.slice(0, -1) : BASE_URL) + '/api';

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
            addBotMessage("❌ Failed to load knowledge base. Please ensure the backend is running.")
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
            }, {
                timeout: 60000
            })

            if (response.status === 200) {
                addBotMessage(response.data.response)
            }
        } catch (error) {
            const errorMsg = error.response?.data?.detail || error.message || 'Failed to get response'
            addBotMessage(`❌ Error: ${errorMsg}`)
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

    if (!dataAvailable) {
        return (
            <div className="tab-panel">
                <h2>🤖 Regulatory Consultant Chatbot</h2>
                <div className="warning" style={{ marginTop: '2rem' }}>
                    <h3>⚠️ No Data Available</h3>
                    <p>Please run the pipeline first to generate the necessary files:</p>
                    <ol style={{ textAlign: 'left', marginTop: '1rem' }}>
                        <li>Upload and process Guideline PDF</li>
                        <li>Run LLM Polishing</li>
                        <li>Upload and process DHF PDF</li>
                        <li>Generate Validation Report</li>
                    </ol>
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
                <div className="chatbot-setup">
                    <div className="info-box">
                        <h3>💡 Load Knowledge Base</h3>
                        <p>Click the button below to index your validation reports and ISO standards.</p>
                        <button
                            className="btn btn-primary"
                            onClick={handleIngestData}
                            disabled={isIngesting}
                            style={{ marginTop: '1rem' }}
                        >
                            {isIngesting ? 'Loading...' : '🔄 Load Knowledge Base'}
                        </button>
                    </div>
                </div>
            ) : (
                <>
                    <div className="chat-messages">
                        {messages.length === 0 && (
                            <div className="welcome-message">
                                <h3>👋 Welcome!</h3>
                                <p>I'm your ISO 11135 regulatory consultant. Ask me anything about your compliance status!</p>
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
