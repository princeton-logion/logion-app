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
import PredictionResultsDisplay from '../components/PredictionResultsDisplay';

function PredictionPage() {
    const [inputText, setInputText] = useState('');
    const [selectedModel, setSelectedModel] = useState('');
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [modelOptions, setModelOptions] = useState([]);
    const { isConnected, sendMessage, addMessageHandler, removeMessageHandler } = useWebSocket();
    const [currentTask, setCurrentTask] = useState(null);
    // task state: { id, status, progress, message, results, error, statusCode }

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
            taskType: "prediction",
            requestData: { model_name: selectedModel, text: inputText },
            pendingMessage: "Submitting word prediction task...",
            taskInProgressMessage: "Word prediction task already in progress."
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

    const isInputValid = inputText.trim() !== '' && inputText.includes('?');
    const predictButtonDisabled = !isInputValid || !isConnected || (currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing'));
    const cancelButtonEnabled = currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing');

    const textareaClasses = `form-control form-control-lg ${isInputValid ? 'is-valid' : ''} ${currentTask?.status === 'error' ? 'is-invalid' : ''}`;
    const predictButtonClass = predictButtonDisabled ? 'btn btn-secondary' : 'btn btn-primary';
    const cancelButtonClass = 'btn btn-danger ms-3';

    // page main content
    return (
        <div>
            {sidebarOpen && <div className="content-overlay" onClick={toggleSidebar}></div>}
            <div className={`main-content ${sidebarOpen ? 'shifted' : ''}`}>
                <div className='container mt-5'>
                    <div className="d-flex align-items-center mb-4">
                        <button className="btn btn-outline-dark me-auto" onClick={toggleSidebar}>☰ Menu</button>
                        <Sidebar sidebarOpen={sidebarOpen} toggleSidebar={toggleSidebar}/>
                        <h1 className="text-center flex-grow-1 m-0">Word Prediction</h1>
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
                                        placeholder='Enter text with "?" for each missing word'
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
                            <PredictionResultsDisplay 
                                taskStatus={currentTask?.status}
                                predictions={predictionsToDisplay}
                                inputText={inputText}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default PredictionPage;