import React from 'react';
import { OverlayTrigger, Popover } from 'react-bootstrap';

const ProbabilityPopover = ({ children, placement = "top", trigger = "click", popoverId = "prob-popover" }) => {
    const popoverContent = (
        <Popover id={popoverId}>
            <Popover.Header as="h3">What is probability?</Popover.Header>
            <Popover.Body>
            The model's predicted likelihood a word appears in the given context.
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

export default ProbabilityPopover;