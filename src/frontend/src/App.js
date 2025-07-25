import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
import PredictionPage from './pages/PredictionPage';
import DetectionPage from './pages/DetectionPage';
import Sidebar from './components/Sidebar';
import { WebSocketProvider } from './contexts/WebSocketContext';
import './App.css';


function App() {

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <WebSocketProvider>
    <Router>
      <div className="App">
        <Sidebar sidebarOpen={sidebarOpen} toggleSidebar={toggleSidebar} />
        <div className={`main-content ${sidebarOpen ? 'shifted' : ''}`}>
          <Routes>
            <Route path="/" element={<MainPage />} />
            <Route path="/prediction" element={<PredictionPage />} />
            <Route path="/detection" element={<DetectionPage />} />
          </Routes>
        </div>
      </div>
    </Router>
    </WebSocketProvider>
  );
}

export default App;
