import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './Sidebar';
import MainPage from './MainPage';
import PredictionPage from './PredictionPage';
import DetectionPage from './DetectionPage';


function App() {

  const [isSidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    setSidebarOpen(!isSidebarOpen);
  };

  return (
    <Router>
      <div className="App">
        <Sidebar isSidebarOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
        <div className={`main-content ${isSidebarOpen ? 'shifted' : ''}`}>
          <Routes>
            <Route path="/" element={<MainPage />} />
            <Route path="/prediction" element={<PredictionPage />} />
            <Route path="/detection" element={<DetectionPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;