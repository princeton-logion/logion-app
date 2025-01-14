import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './MainPage';
import PredictionPage from './PredictionPage';
import DetectionPage from './DetectionPage';
import Sidebar from './Sidebar';


function App() {

  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <Router>
      <div className="App">
        <Sidebar isSidebarOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
        <div className={`main-content ${isSidebarOpen ? 'shifted' : ''}`}> {/*Main content container*/}
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