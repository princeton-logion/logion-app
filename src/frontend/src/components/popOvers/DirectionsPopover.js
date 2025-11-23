import React from 'react';
import '../../App.css';

const DirectionsPopover = ({ isOpen, onClose, pageTitle, pageDirections }) => {
  if (!isOpen) return null;

  const handleOverlayClick = (e) => {
    // close if click outside popover
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleCloseClick = (e) => {
    e.stopPropagation();
    onClose();
  };

  return (
    <div className="directions-popover-overlay" onClick={handleOverlayClick}>
      <div className="directions-popover">
        <div className="directions-popover-header">
          <h2>{pageTitle}</h2>
        </div>
        <div className="directions-popover-body">
          <p>{pageDirections}</p>
        </div>
        <div className="directions-popover-footer">
          <button className="btn btn-primary" onClick={handleCloseClick}>
            Νόμῳ πείθου
          </button>
        </div>
      </div>
    </div>
  );
};

export default DirectionsPopover;