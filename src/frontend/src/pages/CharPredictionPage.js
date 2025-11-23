import React, { useState, useEffect, useRef, useCallback } from 'react';
// import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../App.css';
import Sidebar from '../components/Sidebar';
import { ProgressBar } from 'react-bootstrap';
import '@fortawesome/fontawesome-free/css/all.css';
import { useWebSocket } from '../contexts/WebSocketContext';
import { renderStatusMsg } from '../utils/statusMsgHandler';
import { handleTaskSubmit, handleTaskCancel } from '../utils/taskUtils';
import { processWsMsg } from '../utils/wsMsgUtils';
import CharPredictionResultsDisplay from '../components/CharPredictionResultsDisplay';
import DirectionsPopover from '../components/popOvers/DirectionsPopover';

function CharPredictionPage() {
    const [inputText, setInputText] = useState('');
    const [selectedModel, setSelectedModel] = useState('');
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [modelOptions, setModelOptions] = useState([]);
    const [showDirections, setShowDirections] = useState(false);
    const { isConnected, sendMessage, addMessageHandler, removeMessageHandler } = useWebSocket();
    const [currentTask, setCurrentTask] = useState(null);
    // task state: { id, status, progress, message, results, error, statusCode }

    // first page visit?
    useEffect(() => {
        const hasVisitedCharPrediction = sessionStorage.getItem('hasVisitedCharPred');
        if (!hasVisitedCharPrediction) {
            setShowDirections(true);
            sessionStorage.setItem('hasVisitedCharPred', 'true');
        }
    }, []);

    useEffect(() => {
      fetch("models")
          .then((response) => response.json())
          .then((data) => {
              setModelOptions(data);
              if (data && data.length > 0 && !selectedModel) {
                    setSelectedModel(data[0]);
                }
          })
          .catch((error) => console.error("Unable to fetch models.", error));
    }, [selectedModel]);

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    const handleModelChange = (e) => {
        setSelectedModel(e.target.value);
    };

    const handleCloseDirections = () => {
        setShowDirections(false);
    };

    // WebSocket task msg handler
    const handleWsMsg = useCallback((message) => {
        processWsMsg(message, setCurrentTask);
    }, [setCurrentTask]);


    // msg handler
    useEffect(() => {
        console.log('Register WebSocket message handler.');
        addMessageHandler(handleWsMsg);

        return () => {
            console.log('Deregister WebSocket message handler.');
            removeMessageHandler(handleWsMsg);
            setCurrentTask(null);
        };
    }, [addMessageHandler, removeMessageHandler, handleWsMsg]);

    const taskSubmit = (event) => {
        const predictionOptions = {
            taskType: "char_prediction",
            requestData: { model_name: selectedModel, text: inputText },
            pendingMessage: "Submitting character prediction task...",
            taskInProgressMessage: "Character prediction task already in progress."
        };
        handleTaskSubmit(event, predictionOptions, isConnected, currentTask, setCurrentTask, sendMessage);
    };

    const taskCancel = () => handleTaskCancel(sendMessage, isConnected, currentTask);

    // dynamic render progress bar
    const lastKnownTask = useRef(null);
    useEffect(() => {
        if (currentTask) lastKnownTask.current = currentTask;
    }, [currentTask]);

    let barValue = 0;
    let barColor = undefined;

    if (!isConnected && (!currentTask || currentTask.status !== 'error')) {
        barColor = 'warning';
    } else if (currentTask) {
        switch (currentTask.status) {
            case 'processing':
              barValue = currentTask.progress;
              break;
            case 'success':
              barValue = 100;
              barColor = 'success';
              break;
            case 'error':
              barValue = lastKnownTask.current?.progress || 0;
              barColor = 'danger';
              break;
            case 'cancelled':
              barValue = lastKnownTask.current?.progress || 0;
              barColor = 'danger';
              break;
            default:
              break;
        }
    }

    const predictionsToDisplay = currentTask?.status === 'success' ? currentTask.results?.predictions : null;
    const textToDisplay = currentTask?.status === 'success' ? currentTask.results?.origText : null;

    const isInputValid = inputText.trim() !== '' && inputText.includes('-');
    const predictButtonDisabled = !isInputValid || !isConnected || (currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing'));
    const cancelButtonEnabled = currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing');

    const textareaClasses = `form-control form-control-lg ${isInputValid ? 'is-valid' : ''} ${currentTask?.status === 'error' ? 'is-invalid' : ''}`;
    const predictButtonClass = predictButtonDisabled ? 'btn btn-secondary' : 'btn btn-primary';
    const cancelButtonClass = 'btn btn-danger ms-3';

    // directions popover content
    const directionsTitle = "Logion Character Prediction";
    const directionsMain = (
        <>
        <p style={{ textAlign: 'left', }}>This page generates possible restorations for incomplete texts. To view missing text suggestions, follow these steps:</p>
    <ol style={{ 
            textAlign: 'left', 
            paddingLeft: '40px',
            display: 'inline-block' 
        }}>
    <li><strong>Choose</strong> a model from the dropdown menu</li>
    <li><strong>Enter</strong> your text in the text area</li>
    <li><strong>Type</strong> one dash (<strong>-</strong>) for each missing <span style={{ textDecoration: 'underline' }}>character</span>:
    <div style={{marginLeft: '10px', paddingLeft: '15px', textIndent: '-15px', color: '#555' }}>
        <strong>Ex:</strong> <em>Οὐκ ἐμοῦ, ἀλλὰ τοῦ ----- ἀκούσαντας ὁμολογεῖν σοφόν ἐστιν Ἕν Πάντα εἶναι.</em>
        </div>
        </li>
    <li><strong>Click</strong> "Predict" to generate model suggestions</li>
    </ol>
    
    <p style={{ marginTop: '16px', fontSize: '0.9em', color: 'gray' }}><em>Only use models marked <span style={{ textDecoration: 'underline' }}>Char</span> for character prediction.<br/>Models may take longer to load their first time.</em></p>
    
    
    </>);

    // page main content
    return (
        <div>
            <DirectionsPopover 
                isOpen={showDirections}
                onClose={handleCloseDirections}
                pageTitle={directionsTitle}
                pageDirections={directionsMain}
            />
            {sidebarOpen && <div className="content-overlay" onClick={toggleSidebar}></div>}
            <div className={`main-content ${sidebarOpen ? 'shifted' : ''}`}>
                <div className='container mt-5'>
                    <div className="d-flex align-items-center mb-4">
                        <button className="btn btn-outline-dark me-auto" onClick={toggleSidebar}>☰ Menu</button>
                        <Sidebar sidebarOpen={sidebarOpen} toggleSidebar={toggleSidebar}/>
                        <h1 className="text-center flex-grow-1 m-0">Character Prediction</h1>
                    </div>
                    <div className="d-flex mb-4">
                        <div><p className='inline-label'>Select model: </p>
                            <select className="form-select model-select" value={selectedModel} onChange={handleModelChange}>
                                {modelOptions.map((model, index) => (<option key={index} value={model}>{model}</option>))}
                            </select>
                        </div>
                    </div>
                    <div className="row">
                        <div className="col-md-12">
                            <form onSubmit={taskSubmit}>
                                <div className="mb-3">
                                    <textarea
                                        className={textareaClasses}
                                        rows="4"
                                        style={{ fontSize: '14px', height: '300px' }}
                                        value={inputText}
                                        onChange={(e) => setInputText(e.target.value)}
                                        placeholder='Enter text with "-" for each missing character'
                                    />
                                </div>
                                <div>
                                    <button type="submit" className={predictButtonClass} disabled={predictButtonDisabled}>Predict</button>
                                    <button type="button" className={cancelButtonClass} onClick={taskCancel} disabled={!cancelButtonEnabled}>Cancel</button>
                                </div>
                            </form>
                            <div className="mt-3">
                                <ProgressBar now={barValue} variant={barColor} style={{height: '5px'}}/>
                            </div>
                            {renderStatusMsg(isConnected, currentTask)}
                            <CharPredictionResultsDisplay 
                                taskStatus={currentTask?.status}
                                predictions={predictionsToDisplay}
                                displayText={textToDisplay}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default CharPredictionPage;