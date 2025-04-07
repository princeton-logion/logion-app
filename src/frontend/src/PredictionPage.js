import React, { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import Sidebar from './Sidebar';
import { OverlayTrigger, Popover } from 'react-bootstrap';
import '@fortawesome/fontawesome-free/css/all.css';


function PredictionPage() {
    const [inputText, setInputText] = useState('');
    const [predictions, setPredictions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [successMsg, setSuccessMsg] = useState(null);
    const [selectedModel, setSelectedModel] = useState('Base BERT');
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const isElectron = !!window.electron;
    const [modelOptions, setModelOptions] = useState([]);

    useEffect(() => {
      fetch("http://127.0.0.1:8000/models")
          .then((response) => response.json())
          .then((data) => setModelOptions(data))
          .catch((error) => console.error("Unable to fetch models.", error));
  }, []);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg(null);
    setSuccessMsg(null);


// server connection for dev vs Electron testing
try {
    if (isElectron) {
        // Electron IPC
        const response = await window.electron.ipcRenderer.invoke('predict-request', {
            text: inputText,
            model_name: selectedModel
        });

        if (!response || !response.predictions) {
            throw new Error("Unexpected server response: Missing predictions.");
        }

        setPredictions(response.predictions);
        setSuccessMsg('Εὖγε!<br/>Predictions generated.');
    } else {
        // Axios for external API
        const response = await axios.post(`http://localhost:8000/prediction`, {
            text: inputText,
            model_name: selectedModel
        });

        if (!response.data || !response.data.predictions) {
            throw new Error("Unexpected server response: Missing predictions.");
        }

        setPredictions(response.data.predictions);
        setSuccessMsg('Εὖγε!<br/>Predictions generated.');
    }


} catch (err) {
    setPredictions([]);
    setErrorMsg(err.message);
    console.error("Error:", err);
} finally {
    setLoading(false);
}};


  const isInputValid = inputText.trim() !== '';
  const textareaClasses = `form-control form-control-lg ${isInputValid ? 'is-valid' : ''} ${errorMsg ? 'is-invalid' : ''}`;
  const isButtonDisabled = !isInputValid;
  const buttonClass = isButtonDisabled ? 'btn btn-secondary' : 'btn btn-primary';

// page main content
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
        <button type="submit" className={buttonClass} disabled={isButtonDisabled}>Predict</button>
      </div>
          </form>
          {loading && <p className="text-center text-secondary mt-3"><div className="spinner-border text-secondary me-2" role="status"/>Please wait.<br/>This may take a few seconds.</p>}
          {errorMsg && <p className="text-center text-danger mt-3">λυπούμαι!<br/>{errorMsg}<br/>Please try again.</p>}
          {successMsg && <p className="text-center text-success mt-3" dangerouslySetInnerHTML={{ __html: successMsg }}></p>}
        </div>
        <div className="col-md-4">
            <div className="mt-1">
              {Object.entries(predictions).map(([maskedIndex, prediction]) => (
                <div key={maskedIndex}>
                  <h5>Word Position: {maskedIndex}</h5>
                  <table className="table table-striped">
                    <thead>
                      <tr>
                        <th>Prediction

                        </th>
                        <th>Probability
                        <OverlayTrigger
                            trigger="click"
                            overlay={
                              <Popover id={`popover-probability`}>
                                <Popover.Header as="h3">What is probability?</Popover.Header>
                                <Popover.Body>
                                The model's predicted likelihood of a word appearing in its given context.
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
