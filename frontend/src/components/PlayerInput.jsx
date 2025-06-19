import { useState, useEffect } from 'react';

const PlayerInput = ({ onPredict, onError }) => {
    const [input, setInput] = useState('');
    const [suggestions, setSuggestions] = useState([]);
    const [playerList, setPlayerList] = useState([]);
    
    // fetch player names when component starts
    useEffect(() => {
        const fetchPlayerNames = async() => {
            try {
                const response = await fetch("https://nba-player-predictor.onrender.com/players");
                const data = await response.json();
                setPlayerList(data.players);
            } catch (error) {
                console.error("Failed to load player names", error);
            }
        };

        fetchPlayerNames();
    }, []);

    const handleInputChange = (e) => {
        const value = e.target.value;
        setInput(value);

        // filter suggestions
        if (value.trim() === '') {
            setSuggestions([]);
        } else {
            const filtered = playerList
                .filter(name => name.toLowerCase().includes(value.toLowerCase()))
                .slice(0,5);
            setSuggestions(filtered);
        }
    };


    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim() === '') {
            // if input is empty, call onError with message and return early
            if (onError) onError("Please enter a player name.");
        }

        // clear any pervious errors on valid submit
        if (onError) onError(null);
        
        onPredict(input.trim());
        setSuggestions([]); // clear suggestions on submit
    };

    const handleSuggestionClick = (name) => {
        setInput(name);
        setSuggestions([]);
        if (onError) onError(null);
        onPredict(name);
    };

    return (
        <form onSubmit = {handleSubmit} 
        style = {{ position: 'relative', display: 'flex', gap: '8px', marginBottom: '20px' }}>
            <div>
            <input
                type = "text"
                placeholder = "Search..."
                value = {input}
                onChange = {handleInputChange}
                style = {{
                    flex: 1,
                    padding: '8px',
                    border: 'none',
                    borderBottom: '1px solid',
                    backgroundColor: 'transparent',
                    outline: 'none',
                    fontSize: '16px',
                }}
            />
            {suggestions.length > 0 && (
                <ul style = {{
                    listStyle: 'none',
                    padding: '0',
                    margin: 0,
                    backgroundColor: 'white',
                    position: 'absolute',
                    top: '100%',
                    width: '100%',
                    zIndex: 1000
                    
                }}>
                    {suggestions.map((name, index) => (
                        <li
                            key = {index}
                            onClick = {() => handleSuggestionClick(name)}
                            style = {{
                                fontSize: '12px',
                                color: 'black',
                                padding: '6px 10px',
                                cursor: 'pointer',
                                textAlign: 'left',
                                borderBottom: index !== suggestions.length -1 ? '1px solid #ccc': 'none'
                            }}
                        >
                            {name}
                        </li>
                    ))}
                </ul>
            )}
            </div>
        </form>
    );
};

export default PlayerInput;