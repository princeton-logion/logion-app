import React, { useMemo } from 'react';
import PredictionPopover from './popOvers/PredictionPopover';
import ProbabilityPopover from './popOvers/ProbabilityPopover';

const PredictionResultsDisplay = ({
    taskStatus,
    predictions,
    inputText
}) => {

    const filledTextElements = useMemo(() => {
        if (!predictions) {
            return null;
        }
        
        const topPredictions = Object.values(predictions).map(p => p.predictions[0]?.token || '[?]');
        const textParts = inputText.split('?');

        return textParts.reduce((acc, part, index) => {
            acc.push(<span key={`part-${index}`}>{part}</span>);
            
            if (index < topPredictions.length) {
                const predictionStyle = {
                    color: '#AA4499',
                    fontWeight: 'bold'
                };

                acc.push(
                    <span key={`pred-${index}`} style={predictionStyle}>
                        {`[${topPredictions[index]}]`}
                    </span>
                );
            }
            return acc;
        }, []);
    }, [inputText, predictions]);

    // display
    if (taskStatus !== 'success' || !predictions) {
        if (taskStatus === 'processing' || taskStatus === 'pending') {
            return <p className="text-center text-secondary mt-3"></p>;
        }
        if (taskStatus === 'error') return null;
        return <p className="text-center text-muted mt-3"></p>;
    }

    return (
        <div className="mt-4">
            <h6 className="mb-1 fw-bold">Restored Text</h6>
            <div className="text-highlight-container">
                    {filledTextElements}

            </div>
            
            <div className="mt-4">
                <div className="row">
                {Object.entries(predictions).map(([maskedIndex, prediction]) => (
                     <div className="mb-4" key={maskedIndex}>
                        <h5>Gap Position: {maskedIndex}</h5>
                        <div className="table-responsive">
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
                                    {prediction.predictions.map((pred, i) => (
                                        <tr key={i}>
                                            <td>{pred.token}</td>
                                            <td>{(pred.probability * 100).toFixed(2)}%</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                ))}
                </div>
            </div>
        </div>
    );
};

export default PredictionResultsDisplay;