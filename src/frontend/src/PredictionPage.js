import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import Sidebar from './Sidebar';
import { OverlayTrigger, Popover } from 'react-bootstrap';
import '@fortawesome/fontawesome-free/css/all.css';
const { ipcRenderer } = window.require('electron');


function PredictionPage() {
    const [inputText, setInputText] = useState('');
    const [predictions, setPredictions] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [selectedOption, setSelectedOption] = useState('Base BERT'); /* Default Base BERT Selection */
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);


    const options = [
    'Base BERT',
    'Base ELECTRA',
  ];

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const handleOptionChange = (e) => {
    setSelectedOption(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setSuccess(null);


      try {
        // call to main process via IPC
        const response = await ipcRenderer.invoke('predict-request', {
           text: inputText,
          model_name: selectedOption
        });


      // response validation
      if (!response || !response.predictions) {
        throw new Error("Unexpected response format from the server: Missing predictions.");
      }

        setPredictions(response.predictions);
        setSuccess('Εὖγε!<br/>Predictions generated.');
      } catch (err) {
      setPredictions([]);
        setError(err.message);
      console.error("Error submitting text:", err);
    } finally {
      setIsLoading(false);
      }
    };


  const isInputValid = inputText.trim() !== '';
  const textareaClasses = `form-control form-control-lg ${isInputValid ? 'is-valid' : ''} ${error ? 'is-invalid' : ''}`;
  const isButtonDisabled = !isInputValid;
  const buttonClass = isButtonDisabled ? 'btn btn-secondary' : 'btn btn-primary';

  return (
    <div>

      {/* Content Overlay */}
      {isSidebarOpen && <div className="content-overlay" onClick={toggleSidebar}></div>}

      {/* Page Contents */}
      <div className={`main-content ${isSidebarOpen ? 'shifted' : ''}`}>
        <div className='container mt-5'>
        <div className="d-flex align-items-center mb-4">
        <button className="btn btn-outline-dark me-auto" onClick={toggleSidebar}>
            ☰ Menu
          </button>
          <Sidebar isSidebarOpen={isSidebarOpen} toggleSidebar={toggleSidebar}/>
      <h1 className="text-center flex-grow-1 m-0">Gap Prediction</h1>
      </div>
      <div className="d-flex mb-4">
      <div><p className='inline-label'>Select model: </p>
      <select className="form-select model-select" value={selectedOption} onChange={handleOptionChange}>
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
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
          placeholder="Enter text with [MASK]"
        />
            </div>
            <div>
        <button type="submit" className={buttonClass} disabled={isButtonDisabled}>Predict</button>
      </div>
          </form>
          {isLoading && <p className="text-center text-secondary mt-3"><div className="spinner-border text-secondary me-2" role="status"/>Loading...</p>}
          {error && <p className="text-center text-danger mt-3">λυπούμαι!<br/>{error}<br/>Please try again.</p>}
          {success && <p className="text-center text-success mt-3" dangerouslySetInnerHTML={{ __html: success }}></p>}
        </div>
        <div className="col-md-4">
          {predictions && Object.entries(predictions).length > 0 && ( // check for null object
            <div className="mt-1">
              {Object.entries(predictions).map(([maskedIndex, prediction]) => (
                <div key={maskedIndex}>
                  <h5>Masked Token at index: {maskedIndex}</h5>
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
          )}
        </div>
      </div>
    </div>
    </div>
    </div>
  );
}

export default PredictionPage;