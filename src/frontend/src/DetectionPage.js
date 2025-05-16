import React, { useState, useEffect, useRef, useCallback } from 'react';
// import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import Sidebar from './Sidebar';
import { OverlayTrigger, Popover, ProgressBar } from 'react-bootstrap';
import '@fortawesome/fontawesome-free/css/all.css';
import { useWebSocket } from './contexts/WebSocketContext';

// create unique task IDs
function genID() {
    return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
        (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
    );
}

function DetectionPage() {
    const [inputText, setInputText] = useState('');
    const [selectedModel, setSelectedModel] = useState('');
    const [selectedLevDist, setSelectedLevDist] = useState(1);
    const [activePopoverWord, setActivePopoverWord] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const popoverRef = useRef(null);
    const wordRef = useRef(null);
    // const isElectron = !!window.electron;
    const [modelOptions, setModelOptions] = useState([]);
    const { isConnected, sendMessage, addMessageHandler, removeMessageHandler } = useWebSocket();

    // task state: { id, status, progress, message, results, error }
    const [currentTask, setCurrentTask] = useState(null);

    useEffect(() => {
      fetch("http://127.0.0.1:8000/models")
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

    // detection task msg handler
    const handleDetectMsg = useCallback((message) => {
        console.log("DetectionPage received message:", message);

        // update state per task ID + msg type
        setCurrentTask(prevTask => {
            // only process msg if it matches current task ID
            if (!prevTask || prevTask.id !== message.task_id) {
                return prevTask;
            }
            switch (message.type) {
                case 'ack':
                    return { ...prevTask, status: 'processing'};
                case 'progress':
                    return { ...prevTask, status: 'processing', progress: message.percentage, message: message.message };
                case 'result':
                    return { ...prevTask, status: 'success', results: message.data, progress: 100};
                case 'error':
                    return { ...prevTask, status: 'error', error: message.detail};
                case 'cancelled':
                    return { ...prevTask, status: 'cancelled'};
                default:
                    console.warn(`Unknown message type: ${message.type}`);
                    return prevTask;
            }
        });
    }, []);


    // --- Effect to Register/Unregister Message Handler ---
    useEffect(() => {
        console.log('DetectionPage: Registering message handler.');
        addMessageHandler(handleDetectMsg);
        return () => {
            console.log('DetectionPage: Unregistering message handler.');
            removeMessageHandler(handleDetectMsg);
            setCurrentTask(null); // Clear task state on unmount
        };
    }, [addMessageHandler, removeMessageHandler, handleDetectMsg]); // Stable dependencies


    // results color coding logic
    const setWordColor = (score) => {
        const logScore = Math.log10(score);
        const normalizedScore = Math.max(0, Math.min(1, (logScore + 4) / 4));
        // color-blind-friendly colors
        const colors = [
            '#D55E00', // red
            '#EE7733', // orange
            '#DDAA33', // yellow
            '#228833', // green
        ];
        // map score to color indx
        const colorIndex = Math.floor(normalizedScore * (colors.length - 1));
        return colors[colorIndex];
    };

    const handleWordClick = (wordData) => {
        if (activePopoverWord?.originalIndex === wordData.originalIndex) {
            // click word to open/close pop-up
            setActivePopoverWord(null);
        } else {
            setActivePopoverWord(wordData);
        }
        };

        useEffect(() => {
            const handleClickOutside = (event) => {
            if (popoverRef.current && !popoverRef.current.contains(event.target) && wordRef.current && !wordRef.current.contains(event.target)) {
                setActivePopoverWord(null);
    
            }
            };
    
            document.addEventListener("mousedown", handleClickOutside);
            return () => {
            document.removeEventListener("mousedown", handleClickOutside);
            };
        }, [popoverRef]);


    	// text results renderer
        const renderTextResultsWithColor = () => {
            const predictions = currentTask?.status === 'success' ? currentTask.results?.predictions : [];
            const ccrValues = currentTask?.status === 'success' ? currentTask.results?.ccr : [];

            if (!predictions || predictions.length === 0 || !ccrValues || ccrValues.length === 0) return null;
    
            // zip preds and ccrs to associate correct values
            const textElements = predictions.map((wordPrediction, index) => {
            const { original_word, suggestions } = wordPrediction;
            const ccr = ccrValues[index]?.ccr_value;
            const color = setWordColor(ccr);
    
            return (
                <span
                    key={`${original_word}-${index}`}
                    style={{ color: color, cursor: "pointer" }}
                    onClick={() => handleWordClick({ word: original_word, suggestions, originalIndex: index })}
                    ref={wordRef}
                >
                    {original_word + " "}
                </span>
            );
        });
    
        const currentWordCCR = activePopoverWord && typeof activePopoverWord.originalIndex === 'number' && ccrValues[activePopoverWord.originalIndex]
                ? ccrValues[activePopoverWord.originalIndex].ccr_value
                : null;
    
            return (
            <div className='d-flex'>
                <div className="text-highlight-container col-md-7">
                {textElements}
                </div>
                {activePopoverWord && activePopoverWord.suggestions && (
                <div className="col-md-4">
                    <h5>Suggestions for: {activePopoverWord.word}</h5>
                    <table className="table table-striped">
                    <thead>
                        <tr>
                        <th>Prediction<OverlayTrigger
                               trigger="click"
                               overlay={
                                 <Popover id={`popover-probability`}>
                                   <Popover.Header as="h3">About predictions</Popover.Header>
                                   <Popover.Body>
                                    All Logion predictions lack diacritics.
                                   </Popover.Body>
                                 </Popover>
                               }
                               >
                                <sup>
                             <i className="fas fa-info-circle ms-1" style={{ fontSize: '1em', cursor: 'pointer' }}></i>
                             </sup>
                           </OverlayTrigger></th>
                        <th>Chance-confidence
                        <OverlayTrigger
                               trigger="click"
                               overlay={
                                 <Popover id={`popover-probability`}>
                                   <Popover.Header as="h3">What is CCR?</Popover.Header>
                                   <Popover.Body>
                                   The chance-confidence score measures the probability a word is a mistranscription. A lower score suggests higher error probability.
                                   </Popover.Body>
                                 </Popover>
                               }
                               >
                                <sup>
                             <i className="fas fa-info-circle ms-1" style={{ fontSize: '1em', cursor: 'pointer' }}></i>
                             </sup>
                           </OverlayTrigger>
                        </th>
                        </tr>
                    </thead>
                    <tbody>
                        {activePopoverWord.suggestions.map((pred, idx) => (
                        <tr key={idx}>
                            <td>{pred.token}</td>
                            <td>{typeof currentWordCCR === 'number' ? currentWordCCR.toFixed(4): 'N/A'}</td>
                        </tr>
                        ))}
                    </tbody>
                    </table>
                </div>
                )}
            </div>
            );
        };


   // task handler
    const handleSubmit = (e) => {
        e.preventDefault();

        if (!isConnected) {
            console.error("Not connected to server.");
             setCurrentTask({
                id: 'submit-error', status: 'error', error: 'Not connected to server.', progress: 0, message: '', results: null
            });
            return;
        }

        // check whether task is running
        if (currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing')) {
            console.warn("Error detection task already in progress.");
            return;
        }

        // create unique task ID
        const taskId = genID();
        const request_data = {
            text: inputText,
            model_name: selectedModel,
            lev_distance: selectedLevDist,
        };
        const message = {
            type: "start_detection",
            task_id: taskId,
            request_data: request_data,
        };

        // set task state
        setCurrentTask({
            id: taskId,
            status: 'pending',
            progress: 0,
            message: 'Submitting error detection task...',
            results: null,
            error: null
        });

        console.log(`${JSON.stringify(message)}`);
        sendMessage(message);
    };

    // cancellation process
    const handleCancel = () => {
        if (!currentTask || !isConnected) {
            console.error("Cancel not available.");
            return;
        }
        if (currentTask.status !== 'processing' && currentTask.status !== 'pending') {
            console.warn(`Cannot cancel task ${currentTask.id} in state ${currentTask.status}.`);
            return;
        }
        const message = {
            type: "cancel_task",
            task_id: currentTask.id,
        };
        console.log(`Cancel task ${currentTask.id}...`);
        sendMessage(message);
    };


     // dynamic render status message
     const renderStatusMsg = () => {
      if (!isConnected && (!currentTask || currentTask.status !== 'error')) {
          return <p className="text-center text-warning mt-3">Lost the oracle.</p>;
      }
  
      if (!currentTask) return null;

      switch (currentTask.status) {
          case 'processing':
              return <p className="text-center text-secondary mt-3"> <div className="spinner-border text-secondary spinner-border-sm me-2" role="status"/> {currentTask.message} ({currentTask.progress?.toFixed(1)}%) </p> ;
          case 'success':
               return <p className="text-center text-success mt-3" dangerouslySetInnerHTML={{ __html: 'Εὖγε!' }}></p>;
          case 'error':
              return <p className="text-center text-danger mt-3">λυπούμαι!<br/>{currentTask.error}<br/>Please try again.</p>;
           case 'cancelled':
              return <p className="text-center text-danger mt-3">Error detection cancelled.</p>;
          default:
              return null;
      }
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
    const cancelButtonEnabled = currentTask && isConnected && (currentTask.status === 'pending' || currentTask.status === 'processing' || currentTask.status === 'cancelling'); // Enable only if processing/pending and connected

    const textareaClasses = `form-control form-control-lg ${isInputValid ? 'is-valid' : ''} ${currentTask?.status === 'error' ? 'is-invalid' : ''}`;
    const detectButtonClass = detectButtonDisabled ? 'btn btn-secondary' : 'btn btn-primary';
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
                        <h1 className="text-center flex-grow-1 m-0">Error Detection</h1>
                    </div>
                    <div className="d-flex mb-4 col-md-7">
                        <div><p className='inline-label'>Select model: </p>
                            <select className="form-select model-select" value={selectedModel} onChange={handleModelChange} disabled={!modelOptions.length}>
                                {modelOptions.map((model, index) => (<option key={index} value={model}>{model}</option>))}
                            </select>
                        </div>
                        <div className='ms-3'><p className='inline-label'>Levenshtein distance: <OverlayTrigger
                           trigger="click"
                           placement="bottom"
                           overlay={
                             <Popover id={`popover-probability-levdist`}>
                               <Popover.Header as="h3">What is Levenshtein distance?</Popover.Header>
                               <Popover.Body>
                               The minimum number of edits required to transform one word into another.
                               </Popover.Body>
                             </Popover>
                           }
                           >
                            <sup>
                         <i className="fas fa-info-circle" style={{ fontSize: '1em', cursor: 'pointer' }}></i>
                         </sup>
                       </OverlayTrigger></p>
                            <select className="form-select lev-distance-select" value={selectedLevDist} onChange={handleLevDistChange}>
                                {levDistOptions.map((dist) => (<option key={dist} value={dist}>{dist}</option>))}
                            </select>
                        </div>
                    </div>
                    <div className="row">
                        <div className="col-md-12">
                            <form onSubmit={handleSubmit}>
                                <div className="mb-3">
                                    <textarea
                                        className={textareaClasses}
                                        rows="4" style={{ fontSize: '14px', height: '300px' }} value={inputText} onChange={(e) => setInputText(e.target.value)} placeholder="Enter text" />
                                </div>
                                <div>
                                    <button type="submit" className={detectButtonClass} disabled={detectButtonDisabled}>
                                        Detect Errors
                                    </button>
                                    <button type="button" className={cancelButtonClass} onClick={handleCancel} disabled={!cancelButtonEnabled} >
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
                            {renderStatusMsg()}
                             <div className="mt-3">
                                {renderTextResultsWithColor()}
                             </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default DetectionPage;
