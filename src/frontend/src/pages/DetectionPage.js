import React, { useState, useEffect, useRef, useCallback } from 'react';
// import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../App.css';
import Sidebar from '../components/Sidebar';
import { OverlayTrigger, Popover, ProgressBar } from 'react-bootstrap';
import '@fortawesome/fontawesome-free/css/all.css';
import { useWebSocket } from '../contexts/WebSocketContext';
import { renderStatusMsg } from '../utils/statusMsgHandler';
import { handleTaskSubmit, handleTaskCancel } from '../utils/taskUtils'
import { processWsMsg } from '../utils/wsMsgUtils'
import DetectionResultsDisplay from '../components/DetectionResultsDisplay';
import LevDistPopover from '../components/popOvers/LevDistPopover'
import DirectionsPopover from '../components/popOvers/DirectionsPopover';

function DetectionPage() {
    const [inputText, setInputText] = useState('');
    const [selectedModel, setSelectedModel] = useState('');
    const [selectedLevDist, setSelectedLevDist] = useState(1);
    const [activePopoverWord, setActivePopoverWord] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const popoverRef = useRef(null);
    const wordRef = useRef(null);
    const [modelOptions, setModelOptions] = useState([]);
    const [showDirections, setShowDirections] = useState(false);
    const { isConnected, sendMessage, addMessageHandler, removeMessageHandler } = useWebSocket();
    const [currentTask, setCurrentTask] = useState(null);
    // task state: { id, status, progress, message, results, error }

    // first page visit?
    useEffect(() => {
        const hasVisitedErrDetect = sessionStorage.getItem('hasVisitedErrDetect');
        if (!hasVisitedErrDetect) {
            setShowDirections(true);
            sessionStorage.setItem('hasVisitedErrDetect', 'true');
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

    const levDistOptions = [1, 2, 3];

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    const handleModelChange = (e) => {
        setSelectedModel(e.target.value);
    };

    const handleLevDistChange = (e) => {
       setSelectedLevDist(parseInt(e.target.value, 10));
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
        const detectionOptions = {
            taskType: "detection",
            requestData: {
                text: inputText,
                model_name: selectedModel,
                lev_distance: selectedLevDist,
            },
            pendingMessage: "Submitting error detection task...",
            taskInProgressMessage: "Error detection task already in progress."
        };

        handleTaskSubmit(
            event,
            detectionOptions,
            isConnected,
            currentTask,
            setCurrentTask,
            sendMessage
        );
    };

    const taskCancel = () => {
        handleTaskCancel(sendMessage, isConnected, currentTask);
    };


    // dynamic render progress bar
    const lastKnownTask = useRef(null);
    useEffect(() => {
      if (currentTask) {
        lastKnownTask.current = currentTask;
      }
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
              barValue = lastKnownTask.current;
              barColor = 'danger';
              break;
           case 'cancelled':
              barValue = lastKnownTask.current;
              barColor = 'danger';
              break;
      }
    }


    const isInputValid = inputText.trim() !== '';
    const detectButtonDisabled = !isInputValid || !isConnected || (currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing' || currentTask.status === 'cancelling'));
    const cancelButtonEnabled = currentTask && isConnected && (currentTask.status === 'pending' || currentTask.status === 'processing' || currentTask.status === 'cancelling');

    const textareaClasses = `form-control form-control-lg ${isInputValid ? 'is-valid' : ''} ${currentTask?.status === 'error' ? 'is-invalid' : ''}`;
    const detectButtonClass = detectButtonDisabled ? 'btn btn-secondary' : 'btn btn-primary';
    const cancelButtonClass = 'btn btn-danger ms-3';

    // directions popover content
    const directionsTitle = "Logion Error Detection";
    const directionsMain = (
        <>
        <p style={{ textAlign: 'left', }}>This page generates possible emendations for complete texts. To view text emendations, follow these steps:</p>
    <ol style={{ 
            textAlign: 'left', 
            paddingLeft: '40px',
            display: 'inline-block' 
        }}>
    <li><strong>Choose</strong> a model from the dropdown menu</li>
    <li><strong>Choose</strong> a Levenshtein distance from the dropdown menu</li>
    <li><strong>Enter</strong> your <span style={{ textDecoration: 'underline' }}>complete</span> text in the text area</li>
    <div style={{marginLeft: '10px', paddingLeft: '15px', textIndent: '-15px', color: '#555' }}>
        <strong>Ex:</strong> <em>οὐκ ἐμοῦ ἀλλὰ τοῦ λόγου ἀκούσαντας ὁμολογεῖν σοφόν ἐστιν ἓν πάντα εἶναι</em>
        </div>
    <li><strong>Click</strong> "Detect Errors" to generate model suggestions</li>
    </ol>
    
    <p style={{ marginTop: '16px', fontSize: '0.9em', color: 'gray' }}><em>Models may take longer to load their first time.</em></p>
    
    
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
                        <h1 className="text-center flex-grow-1 m-0">Error Detection</h1>
                    </div>
                    <div className="d-flex mb-4 col-md-7">
                        <div><p className='inline-label'>Select model: </p>
                            <select className="form-select model-select" value={selectedModel} onChange={handleModelChange} disabled={!modelOptions.length}>
                                {modelOptions.map((model, index) => (<option key={index} value={model}>{model}</option>))}
                            </select>
                        </div>
                        <div className='ms-3'><p className='inline-label'>Levenshtein distance: <LevDistPopover>
                            <sup>
                         <i className="fas fa-info-circle ms-1" style={{ fontSize: '1em', cursor: 'pointer' }}></i>
                         </sup>
                       </LevDistPopover></p>
                            <select className="form-select lev-distance-select" value={selectedLevDist} onChange={handleLevDistChange}>
                                {levDistOptions.map((dist) => (<option key={dist} value={dist}>{dist}</option>))}
                            </select>
                        </div>
                    </div>
                    <div className="row">
                        <div className="col-md-12">
                            <form onSubmit={taskSubmit}>
                                <div className="mb-3">
                                    <textarea
                                        className={textareaClasses}
                                        rows="4" style={{ fontSize: '14px', height: '300px' }} value={inputText} onChange={(e) => setInputText(e.target.value)} placeholder="Enter text" />
                                </div>
                                <div>
                                    <button type="submit" className={detectButtonClass} disabled={detectButtonDisabled}>
                                        Detect Errors
                                    </button>
                                    <button type="button" className={cancelButtonClass} onClick={taskCancel} disabled={!cancelButtonEnabled} >
                                             Cancel
                                         </button>
                                </div>
                            </form>

                             <div className="mt-3">
                             <ProgressBar
                 now={barValue}
                 variant={barColor}
                 style={{height: '5px'}}
             />
                             </div>
                            {renderStatusMsg(isConnected, currentTask)}
                             <div className="mt-3">
                                {currentTask && (
                <DetectionResultsDisplay
                    taskStatus={currentTask.status}
                    predictions={currentTask.results?.predictions}
                    ccrValues={currentTask.results?.ccr}
                />
            )}
                             </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default DetectionPage;
