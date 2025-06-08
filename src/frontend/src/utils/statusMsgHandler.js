export const handleErrorMsg = (statusCode, detail) => {
    let baseMessage = `λυπούμαι!`;
    let explanation = "";
    // connection error
    if (detail && detail.toLowerCase().includes("not connected to server")) {
        explanation = "Unable to connect to server. Please check your internet connection.";
    } else if (statusCode) {
        switch (statusCode) {
            // validation error
            case 422:
                explanation = "Input is ill-formed.";
                // input text error
                if (detail && detail.toLowerCase().includes("invalid request data")) {
                    explanation = `Ώπα! Text is improperly formatted.<br/>Please correct and try again.`;
                // input model error
                } else if (detail && detail.toLowerCase().includes("model") && detail.toLowerCase().includes("not available")) {
                    explanation = `${detail.match(/Model '(.*?)'/)?.[1] || ''} currently unavailable.<br/>Please choose another model and try again.`;
                // input Lev filter error
                } else if (detail && detail.toLowerCase().includes("levenshtein filter")) {
                     explanation = `The selected Levenshtein distance is currently unavailable.<br/>Please choose another distance and try again.`;
                }
                else if (detail) {
                    explanation += `<br/>${detail}`;
                }
                break;
            // server error
            case 500:
                explanation = "An unexpected error occurred on the server.";
                if (detail && detail.toLowerCase().includes("unable to load model")) {
                    explanation += `Unable to load ${detail.match(/Model '(.*?)'/)?.[1] || ''}.<br/>Please try a different model.<br/>If the issue persists, contact support.`;
                } else if (detail && detail.toLowerCase().includes("unable to load levenshtein filter")) {
                    explanation += `Unable to load the selected Levenshtein filter.<br/>Please try a different Levenshtein distance.<br/>If the issue persists, contact support.`;
                } else if (detail) {
                    explanation += `<br/>${detail}`;
                }
                break;
            // service error
            case 503:
                explanation = "Cannot access remote resources.";
                if (detail) explanation += `${detail}<br/>Please contact support or try again later.`;
                break;
            // default error
            default:
                explanation = detail || "An unexpected error occurred.";
                break;
        }
    // fallback error
    } else {
        explanation = detail || "An unknown error occurred.";
    }
    return `${baseMessage}<br/>${explanation}`;
};



// dynamic render status message
export const renderStatusMsg = (isConnected, currentTask) => {
    if (!isConnected && (!currentTask || currentTask.status !== 'error')) {
        return <p className="text-center text-warning mt-3">Disconnected.</p>;
    }

    if (!currentTask) return null;

    switch (currentTask.status) {
        case 'processing':
            return <p className="text-center text-secondary mt-3"> <div className="spinner-border text-secondary spinner-border-sm me-2" role="status"/> {currentTask.message} ({currentTask.progress?.toFixed(1)}%) </p> ;
        case 'success':
            return <p className="text-center text-success mt-3" dangerouslySetInnerHTML={{ __html: 'Εὖγε!' }}></p>;
        case 'error':
            const errorMsg = handleErrorMsg(currentTask.statusCode, currentTask.error)
            return <p className="text-center text-danger mt-3" dangerouslySetInnerHTML={{ __html: errorMsg }}></p>;
        case 'cancelled':
            return <p className="text-center text-danger mt-3">Task cancelled.</p>;
        default:
            return null;
    }
};