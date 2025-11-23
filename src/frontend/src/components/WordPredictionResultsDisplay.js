import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import PredictionPopover from './popOvers/PredictionPopover';
import ProbabilityPopover from './popOvers/ProbabilityPopover';

const WordPredictionResultsDisplay = ({
    taskStatus,
    predictions,
    displayText
}) => {
    // track current selected character
    const [activePredictionKey, setActivePredictionKey] = useState(null);
    
    const popoverRef = useRef(null);
    const textRef = useRef(null);

    // reset w/ new predictions
    useEffect(() => {
        setActivePredictionKey(null);
    }, [predictions]);

    // click for popover
    const handleCharClick = useCallback((e, key) => {
        // don't close popover immediately
        e.stopPropagation();
        if (activePredictionKey === key) {
            setActivePredictionKey(null);
        } else {
            setActivePredictionKey(key);
        }
    }, [activePredictionKey]);

    // click elsewhere to close
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (
                popoverRef.current &&
                !popoverRef.current.contains(event.target) &&
                textRef.current &&
                !textRef.current.contains(event.target)
            ) {
                setActivePredictionKey(null);
            }
        };

        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, []);

    const filledTextElements = useMemo(() => {
        if (!predictions) {
            return null;
        }

        const predictionKeys = Object.keys(predictions);
        const topPredictions = predictionKeys.map(key => ({
            key: key,
            token: predictions[key].predictions[0]?.token || '[-]'
        }));

        const textParts = displayText.split('-');
        const elements = [];

        let predIndex = 0;

        for (let i = 0; i < textParts.length; i++) {
            if (textParts[i]) {
                elements.push(<span key={`part-${i}`}>{textParts[i]}</span>);
            }

            if (predIndex < topPredictions.length) {
                const consecutiveBatch = [topPredictions[predIndex]];
                predIndex++;

                while (i + 1 < textParts.length &&
                       textParts[i + 1] === '' &&
                       predIndex < topPredictions.length) {
                    consecutiveBatch.push(topPredictions[predIndex]);
                    predIndex++;
                    i++; // skip empty txt part
                }

                const wrapperStyle = {
                    color: '#AA4499',
                    fontWeight: 'bold'
                };

                // each char individually clickable
                elements.push(
                    <span key={`group-${predIndex}`} style={wrapperStyle}>
                        [
                        {consecutiveBatch.map((item) => {
                            const isSelected = activePredictionKey === item.key;
                            return (
                                <span 
                                    key={item.key}
                                    onClick={(e) => handleCharClick(e, item.key)}
                                    style={{
                                        cursor: 'pointer',
                                        borderBottom: isSelected ? '2px solid #AA4499' : 'none',
                                        backgroundColor: isSelected ? 'rgba(170, 68, 153, 0.1)' : 'transparent'
                                    }}
                                >
                                    {item.token}
                                </span>
                            );
                        })}
                        ]
                    </span>
                );
            }
        }

        return elements;
    }, [displayText, predictions, activePredictionKey, handleCharClick]);

    // display
    if (taskStatus !== 'success' || !predictions) {
        if (taskStatus === 'processing' || taskStatus === 'pending') {
            return <p className="text-center text-secondary mt-3"></p>;
        }
        if (taskStatus === 'error') return null;
        return <p className="text-center text-muted mt-3"></p>;
    }

    const activePredictionData = activePredictionKey ? predictions[activePredictionKey] : null;

    return (
        <div className='row'>
            <div className="col-md-7">
                <h6 className="mb-1 fw-bold">Restored Text</h6>
                {/* this msg is the only difference from CharPredictionResultsDisplay */}
                <small className="text-muted fst-italic text-center d-block mb-2">Click a word in brackets for more information.</small>
                <div className="text-highlight-container" ref={textRef}>
                    {filledTextElements}
                </div>
            </div>

            <div className="col-md-5 d-flex flex-column" style={{ minWidth: 0 }}>
                {activePredictionData && (
                    <div className="mb-3" ref={popoverRef} style={{ width: '100%' }}>
                        <h5>Gap Position: {activePredictionKey}</h5>

                        <table className="table table-striped">
                            <thead>
                                <tr>
                                    <th>
                                        Prediction
                                        <PredictionPopover>
                                            <sup>
                                                <i className="fas fa-info-circle ms-1"></i>
                                            </sup>
                                        </PredictionPopover>
                                    </th>
                                    <th>
                                        Probability
                                        <ProbabilityPopover>
                                            <sup>
                                                <i className="fas fa-info-circle ms-1"></i>
                                            </sup>
                                        </ProbabilityPopover>
                                    </th>
                                </tr>
                            </thead>
                            <tbody>
                                {activePredictionData.predictions.map((pred, i) => (
                                    <tr key={i}>
                                        <td>{pred.token}</td>
                                        <td>{(pred.probability * 100).toFixed(2)}%</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};

export default WordPredictionResultsDisplay;