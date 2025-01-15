import React, { useState, useEffect, useRef } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import Sidebar from './Sidebar';
import { OverlayTrigger, Popover } from 'react-bootstrap';
import '@fortawesome/fontawesome-free/css/all.css';

const { ipcRenderer } = window.require('electron');

function DetectionPage() {
    const [inputText, setInputText] = useState('');
    const [predictions, setPredictions] = useState([]);
    const [ccrValues, setCcrValues] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [selectedOption, setSelectedOption] = useState('Base BERT');
    const [selectedLevDistance, setSelectedLevDistance] = useState(1); // Default to 1
    const [activePopoverWord, setActivePopoverWord] = useState(null);
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const popoverRef = useRef(null);
    const wordRef = useRef(null);

    const options = [
        'Base BERT',
        'Base ELECTRA',
    ];

    const levDistanceOptions = [1, 2, 3]; // Available Lev Distance values

    const toggleSidebar = () => {
        setIsSidebarOpen(!isSidebarOpen);
    };

    const handleOptionChange = (e) => {
        setSelectedOption(e.target.value);
    };

    const handleLevDistanceChange = (e) => {
       setSelectedLevDistance(parseInt(e.target.value, 10)); // Ensure it's a number
    };

    const getColor = (score) => {
        // Use log scale for better visual contrast
        const logScore = Math.log10(score);
        const normalizedScore = Math.max(0, Math.min(1, (logScore + 4) / 4));

        // Normalize and clip the score for the HSL range
        const hue = (120 * normalizedScore).toFixed(0);
        return `hsl(${hue}, 100%, 50%)`;
    };

    const handleWordClick = (word) => {
    if (activePopoverWord?.word === word.word) {
        // Clicking the same word again closes the popover
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


    const renderTextWithHighlights = () => {
        if (!predictions || predictions.length === 0 || !ccrValues || ccrValues.length === 0) return null;
    
    // Zip predictions and ccr values to associate the correct values
        const textElements = predictions.map((wordPrediction, index) => {
        const { original_word, suggestions } = wordPrediction;
        const ccr = ccrValues[index]?.ccr_value;
        const color = getColor(ccr);
    
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
            {/* Table to display suggestions */}
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
       setIsLoading(true);
       setError(null);
       setSuccess(null);

       try {
           // Make a call to the main process via IPC
           const response = await ipcRenderer.invoke('detect-request', {
               text: inputText,
               model_name: selectedOption,
               lev_distance: selectedLevDistance,
           });

            // Robust response validation
            if (!response || !response.predictions || !response.ccr) {
              throw new Error("Unexpected response format from the server: Missing predictions or CCR values.");
            }

            setPredictions(response.predictions);
            setCcrValues(response.ccr);
            setSuccess('Εὖγε!<br/>Predictions generated.');
        } catch (err) {
            setPredictions([]);
            setCcrValues([]);
            setError(err.message);
            console.error("Error submitting the form:", err);
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

            <div className={`main-content ${isSidebarOpen ? 'shifted' : ''}`}>
                <div className='container mt-5'>
                    <div className="d-flex align-items-center mb-4">
                    <button className="btn btn-outline-dark me-auto" onClick={toggleSidebar}>
                        ☰ Menu
                    </button>
                    <Sidebar isSidebarOpen={isSidebarOpen} toggleSidebar={toggleSidebar}/>
                <h1 className="text-center flex-grow-1 m-0">Error Detection</h1>
                </div>
                <div className="d-flex mb-4 col-md-7">
                    <div><p className='inline-label'>Select model: </p>
                    <select className="form-select model-select" value={selectedOption} onChange={handleOptionChange}>
                    {options.map((option) => (
                        <option key={option} value={option}>
                        {option}
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
                    <select className="form-select lev-distance-select" value={selectedLevDistance} onChange={handleLevDistanceChange}>
                    {levDistanceOptions.map((dist) => (
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
                    <button type="submit" className={buttonClass} disabled={isButtonDisabled}>Detect errors</button>
                </div>
                    </form>
                    {isLoading && <p className="text-center text-secondary mt-3"><div className="spinner-border text-secondary me-2" role="status"/>Loading...</p>}
                    {error && <p className="text-center text-danger mt-3">λυπούμαι!<br/>{error}<br/>Please try again.</p>}
                    {success && <p className="text-center text-success mt-3" dangerouslySetInnerHTML={{ __html: success }}></p>}
                    <div>
                    {renderTextWithHighlights()}
                    </div>
                    </div>
                    
                </div>
            </div>
            </div>
        </div>
    );
}

export default DetectionPage;