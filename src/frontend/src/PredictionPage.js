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

function PredictionPage() {
    const [inputText, setInputText] = useState('');
    const [selectedModel, setSelectedModel] = useState('');
    const [sidebarOpen, setSidebarOpen] = useState(false);
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
            .catch((error) => {
                console.error("Cannot access models:", error);
            });
    }, [selectedModel]);


    // prediction task msg handler
    const handlePredictMsg = useCallback((message) => {
        console.log("Received:", message);

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


    // msg handler
    useEffect(() => {
        console.log('Initialize message handler.');
        addMessageHandler(handlePredictMsg);

        return () => {
            console.log('Deregister message handler.');
            removeMessageHandler(handlePredictMsg);
            setCurrentTask(null);
        };
    }, [addMessageHandler, removeMessageHandler, handlePredictMsg]);

    // sidebar toggle
    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };
    
    // model handler
    const handleModelChange = (e) => {
        setSelectedModel(e.target.value);
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
            console.warn("Prediction task already in progress.");
            return;
        }

        // create unique task ID
        const taskId = genID();
        const request_data = {
            model_name: selectedModel,
            text: inputText,
        };
        const message = {
            type: "start_prediction",
            task_id: taskId,
            request_data: request_data,
        };

        // set task state
        setCurrentTask({
            id: taskId,
            status: 'pending',
            progress: 0,
            message: 'Submitting prediction task...',
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
              return <p className="text-center text-danger mt-3">Prediction cancelled.</p>;
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


    // show predictions only once task successful
    const predictionsToDisplay = currentTask?.status === 'success' ? currentTask.results?.predictions : {};


    const isInputValid = inputText.trim() !== '';
    const predictButtonDisabled = !isInputValid || !isConnected || (currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing'));
    const cancelButtonEnabled = currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing');

    const textareaClasses = `form-control form-control-lg ${isInputValid ? 'is-valid' : ''} ${currentTask?.status === 'error' ? 'is-invalid' : ''}`;
    const predictButtonClass = predictButtonDisabled ? 'btn btn-secondary' : 'btn btn-primary';
    const cancelButtonClass = 'btn btn-danger ms-3';


    return (
        <div>
      {sidebarOpen && <div className="content-overlay" onClick={toggleSidebar}></div>}
            <div className={`main-content ${sidebarOpen ? 'shifted' : ''}`}>
                <div className='container mt-5'>
                <div className="d-flex align-items-center mb-4">
        <button className="btn btn-outline-dark me-auto" onClick={toggleSidebar}>
            ☰ Menu
          </button>
          <Sidebar sidebarOpen={sidebarOpen} toggleSidebar={toggleSidebar}/>
      <h1 className="text-center flex-grow-1 m-0">Word Prediction</h1>
      </div>
      <div className="d-flex mb-4">
      <div><p className='inline-label'>Select model: </p>
      <select className="form-select model-select" value={selectedModel} onChange={handleModelChange}>
      {modelOptions.map((model, index) => (
                <option key={index} value={model}>
                    {model}
                </option>
          ))}
        </select></div>

      </div>
                    <div className="row">
                        <div className="col-md-8">
                            <form onSubmit={handleSubmit}>
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
                                    <button type="submit" className={predictButtonClass} disabled={predictButtonDisabled}>
                                        Predict
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
                        </div>
                        <div className="col-md-4">
            <div className="mt-1">
              {Object.entries(predictionsToDisplay || {}).map(([maskedIndex, prediction]) => (
                <div key={maskedIndex}>
                  <h5>Word Position: {maskedIndex}</h5>
                  <table className="table table-striped">
                    <thead>
                      <tr>
                        <th>Prediction
                        <OverlayTrigger
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
                       </OverlayTrigger>
                        </th>
                        <th>Probability
                        <OverlayTrigger
                            trigger="click"
                            overlay={
                              <Popover id={`popover-probability`}>
                                <Popover.Header as="h3">What is probability?</Popover.Header>
                                <Popover.Body>
                                The model's predicted likelihood a word appears in the given context.
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
                      {prediction.predictions.map((pred, i) => (
                        <tr key={i}>
                          <td>{pred.token}</td>
                          <td>{(pred.probability * 100).toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default PredictionPage;