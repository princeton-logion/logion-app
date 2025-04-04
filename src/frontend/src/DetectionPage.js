import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import Sidebar from './Sidebar';
import { OverlayTrigger, Popover } from 'react-bootstrap';
import '@fortawesome/fontawesome-free/css/all.css';

function DetectionPage() {
    const [inputText, setInputText] = useState('');
    const [predictions, setPredictions] = useState([]);
    const [ccrValues, setCcrValues] = useState([]);
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState(null);
    const [successMsg, setSuccessMsg] = useState(null);
    const [selectedModel, setSelectedModel] = useState('Base BERT');
    const [selectedLevDist, setSelectedLevDist] = useState(1);
    const [activePopoverWord, setActivePopoverWord] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const popoverRef = useRef(null);
    const wordRef = useRef(null);
    const isElectron = !!window.electron;
    const [modelOptions, setModelOptions] = useState([]);

    useEffect(() => {
      fetch("http://127.0.0.1:8000/models")
          .then((response) => response.json())
          .then((data) => setModelOptions(data))
          .catch((error) => console.error("Unable to fetch models.", error));
  }, []);

    const levDistOptions = [1, 2, 3];

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };

    const handleModelChange = (e) => {
        setSelectedModel(e.target.value);
    };

    const handleLevDistChange = (e) => {
       setSelectedLevDist(parseInt(e.target.value, 10)); // confirm Lev is number
    };


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

    const handleWordClick = (word) => {
    if (activePopoverWord?.word === word.word) {
        // click word to open/close pop-up
        setActivePopoverWord(null);
    } else {
        setActivePopoverWord(word);
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
                onClick={() => handleWordClick({ word: original_word, suggestions })}
                ref={wordRef}
            >
                {original_word + " "}
            </span>
        );
    });

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
                    <th>Prediction</th>
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
                    {activePopoverWord.suggestions.map((pred, idx) => (
                    <tr key={idx}>
                        <td>{pred.token}</td>
                        <td>{(pred.probability * 100).toFixed(2)}%</td>
                    </tr>
                    ))}
                </tbody>
                </table>
            </div>
            )}
        </div>
        );
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
        const response = await window.electron.ipcRenderer.invoke('detect-request', {
            text: inputText,
            model_name: selectedModel,
            lev_distance: selectedLevDist,
        });

        // response validation
        if (!response || !response.predictions || !response.ccr) {
            throw new Error("Unexpected server response: Missing predictions or CCR scores.");
        }

        setPredictions(response.predictions);
        setCcrValues(response.ccr);
        setSuccessMsg('Εὖγε!<br/>Predictions generated.');
    } else {
        // Axios for external API
        const response = await axios.post(`http://localhost:8000/detection`, {
            text: inputText,
            model_name: selectedModel,
            lev_distance: selectedLevDist,
        });

        // response validation
        if (!response.data || !response.data.predictions || !response.data.ccr) {
            throw new Error("Unexpected server response: Missing predictions or CCR scores.");
        }

        setPredictions(response.data.predictions);
        setCcrValues(response.data.ccr);
        setSuccessMsg('Εὖγε!<br/>Predictions generated.');
    }
} catch (err) {
    setPredictions([]);
    setCcrValues([]);
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
                <h1 className="text-center flex-grow-1 m-0">Error Detection</h1>
                </div>
                <div className="d-flex mb-4 col-md-7">
                    <div><p className='inline-label'>Select model: </p>
                    <select className="form-select model-select" value={selectedModel} onChange={handleModelChange}>
                    {modelOptions.map((model, index) => (
                <option key={index} value={model}>
                    {model}
                </option>
                    ))}
                    </select>
                </div>
                <div><p className='inline-label'>Levenshtein distance: <OverlayTrigger
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
                    {levDistOptions.map((dist) => (
                        <option key={dist} value={dist}>
                        {dist}
                        </option>
                    ))}
                    </select>
                </div>
            </div>
                <div className="row">
                    <div>
                    <form onSubmit={handleSubmit}>
                        <div className="mb-3">
                        <textarea
                    className={textareaClasses}
                    rows="4"
                    style={{ fontSize: '14px', height: '300px' }}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter text"
                    />
                        </div>
                        <div>
                    <button type="submit" className={buttonClass} disabled={isButtonDisabled}>Detect Errors</button>
                </div>
                    </form>
                    {loading && <p className="text-center text-secondary mt-3"><div className="spinner-border text-secondary me-2" role="status"/>Please wait.<br/>This may take several minutes.</p>}
                    {errorMsg && <p className="text-center text-danger mt-3">λυπούμαι!<br/>{errorMsg}<br/>Please try again.</p>}
                    {successMsg && <p className="text-center text-success mt-3" dangerouslySetInnerHTML={{ __html: successMsg }}></p>}
                    <div>
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
