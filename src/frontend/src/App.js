import React, { useState } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
import WordPredictionPage from './pages/PredictionPage';
import CharPredictionPage from './pages/CharPredictionPage';
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
            <Route path="/word_prediction" element={<WordPredictionPage />} />
            <Route path="/char_prediction" element={<CharPredictionPage />} />
            <Route path="/detection" element={<DetectionPage />} />
          </Routes>
        </div>
      </div>
    </Router>
    </WebSocketProvider>
  );
}

export default App;
