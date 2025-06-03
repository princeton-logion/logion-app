import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import { useWebSocket } from '../contexts/WebSocketContext';


function MainPage() {

  const { isConnected } = useWebSocket();

  const [typing, setTyping] = useState(true);
  const [appTitle, setAppTitle] = useState("");
  const fullTitle = "λόγιον";
    const typewriter = () => {
      let i = 0;
      const intervalId = setInterval(() => {
        if (i < fullTitle.length) {
          setAppTitle(fullTitle.substring(0, i + 1));
          i++;

        } else {
          clearInterval(intervalId);
          setTyping(false);
        }
      }, 100);
    };
    useEffect(() => {
      typewriter();
    }, []);


  return (


    <div className="main-content">

      {typing ? (
        <h1 className="logion-title typewriter">
          {appTitle}
        </h1>
      ) : (
        <h1 className="logion-title">
          {fullTitle}
        </h1>
      )}
      <h5 className='p-5'>Welcome to Logion, a tool to aid philological research of premodern Greek texts.</h5>

      {/*<div style={{ position: 'absolute', top: '10px', right: '10px', fontSize: '0.8em' }}>
          Status: {isConnected
            ? <span className="badge bg-success">Connected</span>
            : <span className="badge bg-danger">Disconnected</span>
          }
      </div>*/}

      <Link to="/prediction" className="btn btn-pill">Word Prediction</Link>

      <Link to="/detection" className="btn btn-pill">Error Detection</Link>

      <div>
      {/*<Link to="https://princeton-logion.github.io/logion-app/" className="btn btn-pill mt-4">Help</Link>*/}
      </div>

          </div>


  );
}

export default MainPage;
