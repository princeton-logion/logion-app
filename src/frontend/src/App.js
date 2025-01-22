import React, { useState, useEffect } from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import MainPage from './MainPage';
import PredictionPage from './PredictionPage';
import DetectionPage from './DetectionPage';
import Sidebar from './Sidebar';
import axios from 'axios';


function App() {

    const [isSidebarOpen, setIsSidebarOpen] = useState(false);

    const toggleSidebar = () => {
        setIsSidebarOpen(!isSidebarOpen);
    };

    useEffect(() => {
        if (window.api) {
          window.axios = axios;
            window.api.once("set-axios-base-url", (baseURL) => {
              window.axios.defaults.baseURL = baseURL;
          });
        }
    }, [])

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