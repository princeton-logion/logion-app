export const processWsMsg = (message, setCurrentTask) => {
    console.log("Processing WebSocket message:", message);

    setCurrentTask(prevTask => {
        // only process msg if it matches current task ID
        if (!prevTask || prevTask.id !== message.task_id) {
            return prevTask;
        }
        switch (message.type) {
            case 'ack':
                return { ...prevTask, id: message.task_id, status: 'processing', message: message.message};
            case 'progress':
                return { ...prevTask, id: message.task_id, status: 'processing', progress: message.percentage, message: message.message };
            case 'result':
                return { ...prevTask, id: message.task_id, status: 'success', results: message.data, progress: 100};
            case 'error':
                return { ...prevTask, id: message.task_id, status: 'error', error: message.detail, statusCode: message.status_code};
            case 'cancelled':
                return { ...prevTask, id: message.task_id, status: 'cancelled'};
            default:
                console.warn(`Unknown message type: ${message.type}`);
                return prevTask;
        }
    });
};