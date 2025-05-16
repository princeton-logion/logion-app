import asyncio
import logging

async def check_cancel_status(cancellation_event: asyncio.Event, task_id: str):
    """   
    Checks status of 'cancellation_event' variable.
    If cancellation_event is set, raise asyncio.CancelledError to terminate current task. 
    If cancellation_event not set, yield control to resume current task.
    
    Parameters:
        cancellation_event (asyncio.Event) -- event object signaling task cancellation
        task_id (str) -- task UID
    """
    if cancellation_event.is_set():
        logging.info(f"Task {task_id} canceled by user")
        raise asyncio.CancelledError(f"Task {task_id} canceled.")
    await asyncio.sleep(0)