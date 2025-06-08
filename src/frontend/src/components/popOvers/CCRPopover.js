import React from 'react';
import { OverlayTrigger, Popover } from 'react-bootstrap';

const CCRPopover = ({ children, placement = "top", trigger = "click", popoverId = "ccr-popover" }) => {
    const popoverContent = (
        <Popover id={popoverId}>
            <Popover.Header as="h3">What is CCR?</Popover.Header>
            <Popover.Body>
                The chance-confidence score measures the probability a word is a mistranscription. A lower score suggests higher error probability. See{' '}<a href="https://princeton-logion.github.io/logion-app/explainers/ccr/" target="_blank" rel="noopener noreferrer"> Logion's documentation</a>{' '}for more.
            </Popover.Body>
        </Popover>
    );

    return (
        <OverlayTrigger
            trigger={trigger}
            placement={placement}
            overlay={popoverContent}
            rootClose={true}
        >
            {children}
        </OverlayTrigger>
    );
};

export default CCRPopover;