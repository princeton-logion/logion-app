import React from 'react';
import { OverlayTrigger, Popover } from 'react-bootstrap';

const PredictionPopover = ({ children, placement = "top", trigger = "click", popoverId = "pred-popover" }) => {
    const popoverContent = (
        <Popover id={popoverId}>
            <Popover.Header as="h3">About Logion predictions</Popover.Header>
            <Popover.Body>
                All Logion predictions lack diacritics.
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

export default PredictionPopover;