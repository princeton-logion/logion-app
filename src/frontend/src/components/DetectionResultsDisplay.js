import React, { useState, useEffect, useRef, useCallback } from 'react';
import { OverlayTrigger, Popover } from 'react-bootstrap';
import { handleWordColor } from '../utils/detectResultsUtils';
import PredictionPopover from './popOvers/PredictionPopover';
import CCRPopover from './popOvers/CCRPopover';

const DetectionResultsDisplay = ({
    taskStatus,
    predictions,
    ccrValues
}) => {
    const [activePopoverWord, setActivePopoverWord] = useState(null);
    const popoverRef = useRef(null);
    const wordRef = useRef(null);

    useEffect(() => {
        setActivePopoverWord(null);
    }, [predictions]);

    // click to toggle word popover
    const handleWordClick = useCallback((wordData) => {
        // wordData is { original_word, suggestions, originalIndex }
        if (activePopoverWord?.originalIndex === wordData.originalIndex) {
            setActivePopoverWord(null);
        } else {
            setActivePopoverWord(wordData);
        }
    }, [activePopoverWord]);

    // click outside word to toggle popover
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (
                popoverRef.current &&
                !popoverRef.current.contains(event.target) &&
                wordRef.current &&
                !wordRef.current.contains(event.target)
            ) {
                setActivePopoverWord(null);
            }
        };

        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [popoverRef]);

    // display
    if (taskStatus !== 'success' || predictions.length === 0 || ccrValues.length === 0) {
        if (taskStatus === 'processing' || taskStatus === 'pending') {
            return <p className="text-center text-secondary mt-3"></p>;
        }
        if (taskStatus === 'error') return null;
        return <p className="text-center text-muted mt-3"></p>;
    }

    const coloredTextElements = predictions.map((wordPrediction, index) => {
        const { original_word, suggestions } = wordPrediction;
        const ccrScore = ccrValues[index]?.ccr_value;
        const color = handleWordColor(ccrScore);
        const wordData = { original_word, suggestions, originalIndex: index };

        return (
            <span
                key={`colored-${original_word}-${index}`}
                style={{
                    color: color,
                    cursor: 'pointer',
                    borderBottom: activePopoverWord?.originalIndex === index ? '2px solid #AA4499' : 'none'
                }}
                onClick={() => handleWordClick(wordData)}
            >
                {original_word + " "}
            </span>
        );
    });

    const blackTextElements = predictions.map((wordPrediction, index) => {
        const { original_word, suggestions } = wordPrediction;
        const isSelected = activePopoverWord?.originalIndex === index;
        const replacement = isSelected && suggestions.length > 0 ? suggestions[0].token : null;
        const style = replacement ? {
            color: '#AA4499',
            fontWeight: 'bold',
            borderBottom: '2px solid #AA4499'
        } : { color: 'black' };

        return (
            <span
                key={`black-${original_word}-${index}`}
                style={style}
            >
                {(replacement || original_word) + " "}
            </span>
        );
    });

    const selectedOriginalWord = activePopoverWord?.original_word || '';
    const selectedWordSuggestions = activePopoverWord?.suggestions || [];
    const selectedWordIndex = activePopoverWord?.originalIndex;
    const selectedCCR = (typeof selectedWordIndex === 'number' && ccrValues[selectedWordIndex])
        ? ccrValues[selectedWordIndex].ccr_value
        : null;

    return (
        <div className='d-flex'>
            <div className="col-md-7">
            <h6 className="mb-1 fw-bold">Original Text</h6>
            <small className="text-muted fst-italic text-center d-block mb-2">Click a word to view suggested emendations.</small>
            <div className="text-highlight-container mb-3">
                {coloredTextElements}
            </div>

            <h6 className="mb-1 fw-bold">Suggested Text</h6>
            <div className="text-highlight-container mb-4" style={{ borderTop: '1px solid #ccc', paddingTop: '10px' }}>
                {blackTextElements}
            </div>
            </div>

            {activePopoverWord && selectedWordSuggestions && (
                <div className="col-md-4">
                    <h5>Suggestions for: <strong>{selectedOriginalWord}</strong></h5>
                    <table className="table table-striped">
                        <thead>
                            <tr>
                                <th>
                                    Prediction
                                    <PredictionPopover>
                                        <sup>
                                            <i className="fas fa-info-circle ms-1" style={{ fontSize: '1em', cursor: 'pointer' }}></i>
                                        </sup>
                                    </PredictionPopover>
                                </th>
                                <th>
                                    Chance-confidence
                                    <CCRPopover>
                                        <sup>
                                            <i className="fas fa-info-circle ms-1" style={{ fontSize: '1em', cursor: 'pointer' }}></i>
                                        </sup>
                                    </CCRPopover>
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {selectedWordSuggestions.map((pred, idx) => (
                                <tr key={idx}>
                                    <td>{pred.token}</td>
                                    <td>{typeof selectedCCR === 'number' ? selectedCCR.toFixed(4) : 'N/A'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default DetectionResultsDisplay;
