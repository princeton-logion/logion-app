import { genID } from './genID';

// submission handler
export const handleTaskSubmit = (
    event,
    options,
    isConnected,
    currentTask,
    setCurrentTask,
    sendMessage
) => {
    event.preventDefault();

    const {
        taskType,
        requestData,
        pendingMessage,
        taskInProgressMessage = `${taskType} task already in progress.`
    } = options;

    if (!isConnected) {
        console.error(`Unable to submit ${taskType} task. Not connected to server.`);
        setCurrentTask({
            id: `${taskType}-error-disconnected`,
            status: 'error',
            progress: 0,
            message: '',
            results: null,
            error: 'Not connected to server. Please check your connection and try again.',
            statusCode: null
        });
        return;
    }

    if (currentTask && (currentTask.status === 'pending' || currentTask.status === 'processing')) {
        console.warn(taskInProgressMessage);
        return;
    }

    const taskId = genID();
    const message = {
        type: `start_${taskType}`,
        task_id: taskId,
        request_data: requestData,
    };

    setCurrentTask({
        id: taskId,
        status: 'pending',
        progress: 0,
        message: pendingMessage,
        results: null,
        error: null,
        statusCode: null
    });

    console.log(`Submitting ${taskType} task: ${JSON.stringify(message)}`);
    sendMessage(message);
};



// cancellation handler
export const handleTaskCancel = (sendMessage, isConnected, currentTask) => {
    if (!currentTask || !isConnected) {
        console.error("Cancel unavailable.");
        return;
    }
    if (currentTask.status !== 'processing' && currentTask.status !== 'pending') {
        console.warn(`Unable to cancel task ${currentTask.id}. Task is ${currentTask.status}.`);
        return;
    }
    const message = {
        type: "cancel_task",
        task_id: currentTask.id,
    };
    console.log(`Cancel task ${currentTask.id}...`);
    sendMessage(message);
};