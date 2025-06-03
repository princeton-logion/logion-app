import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import '../App.css';

function Sidebar({ sidebarOpen, toggleSidebar }) {
  const location = useLocation();


  return (
    <div className={`sidebar ${sidebarOpen ? 'open' : ''}`} role='navigation'>
      <div className="sidebar-header">
      <Link className="sidebar-header-content" to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
        <img src="logo192.png" className="sidebar-logo" alt="logo" />
        <div className='sidebar-text'><h2>Logion</h2></div>
        </Link>
      </div>
      <ul className="list-unstyled components">

        <div className='nav-title'>Philological Tasks</div>

        <li className={location.pathname === "/prediction" ? "active" : ""}>
          <Link to="/prediction">Word Prediction</Link>
        </li>

        <li className={location.pathname === "/detection" ? "active" : ""}>
          <Link to="/detection">Error Detection</Link>
        </li>
      </ul>

      <ul className='list-unstyled components mt-5'>
          <li><a href="https://princeton-logion.github.io/logion-app/" target="_blank" rel="noopener noreferrer">Help</a></li>
          <li><a href='https://www.logionproject.princeton.edu/' target="_blank" rel="noopener noreferrer">About</a></li>
      </ul>

      <button className="btn btn-close btn-close-white" onClick={toggleSidebar}></button>

    </div>
  );
}

export default Sidebar;
