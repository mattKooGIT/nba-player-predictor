import { useState } from 'react'
import './App.css'
import PlayerInput from './components/PlayerInput.jsx'
import PredictionDisplay from './components/PredictionDisplay.jsx';

function App() {
  const [count, setCount] = useState(0)

  // add error state
  const [error, setError] = useState(null);
  const [predictionData, setPredictionData] = useState(null);

  const handlePredict = async (playerName) => {
  try {
    const response = await fetch("https://nba-player-predictor.onrender.com/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ player_name: playerName }),
    });

    const data = await response.json();

    if (data.error) {
      // if backend returns an error, set error state and clear prediction
      setError(data.error);
      setPredictionData(null);
    }
    else {
      // clear error and set prediction data
      setError(null);
      setPredictionData(data);
    }
  } catch (error) {
    console.error("Error fetching prediction:", error)
    setError("Failed to fetch prediction. Please try again.");
    setPredictionData(null);
  }
};

  return (
    <>
      <title>NBA Stat Line Predictor</title>
      <div style = {{ display: 'flex', alignItems: 'center', gap: '12px', justifyContent: 'center' }}>
        <div style = {{ textAlign: 'right'}}>
          <h1 style = {{ margin: 0, marginTop: '0px'}}>NBA Stat Line<br></br> Predictor</h1>
          <p style = {{ marginTop: '4px', fontWeight: 100, fontSize: '20px' }}>
            by Matt Koo
          </p>
        </div>
        <a href="https://nba.com" target="_blank">
          <img 
              src = "/assets/nba_logo.png"
              className = "nba logo" 
              alt = "NBA logo"
              style = {{ width: '100px', height: 'auto'}} />
        </a>
      </div>
      <div className = "search-caption">
        <p style = {{ fontStyle: 'italic', fontWeight: 100 }}>
          Enter an NBA Player from the 2024-2025 season
        </p>
        <p style = {{ fontStyle: 'italic', fontWeight: 200, fontSize: '13px' }}>
          ***Please wait 10-30 seconds for backend to load :D***
        </p>
      </div>
      <div className = "search-bar">
        <PlayerInput onPredict = {handlePredict} onError = {setError} />
      </div>
      
      {/* Show error message if exists */}
      {error && (
        <div style = {{ color: 'red', marginBottom: '16px', textAlign: 'center'}}>
          {error}
        </div>
      )}
      <div className = "prediction-data">
        <PredictionDisplay predictionData = {predictionData}/>
      </div>
    </>
  )
}

export default App
