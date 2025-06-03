import React from 'react';
import { OverlayTrigger, Popover } from 'react-bootstrap';

const ProbabilityPopover = ({ children, placement = "bottom", trigger = "click", popoverId = "lev-dist-popover" }) => {
    const popoverContent = (
        <Popover id={popoverId}>
            <Popover.Header as="h3">What is Levenshtein distance?</Popover.Header>
            <Popover.Body>
            The minimum number of edits required to transform one word into another. See{' '}<a href="https://princeton-logion.github.io/logion-app/explainers/lev-dist/" target="_blank" rel="noopener noreferrer"> Logion's documentation</a>{' '}for more.
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