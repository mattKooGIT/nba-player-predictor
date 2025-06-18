const PredictionDisplay = ({ predictionData }) => {
    if (!predictionData) return null;

    const { player_name, headshot_url, predictions } = predictionData;

    // stat order
    const statOrder = [
        'FG_PCT',
        "FT_PCT",
        "FG3M",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
    ]
    // convert and format predictions
    const formattedPredictions = statOrder
        .filter(stat => stat in predictions)
        .map(stat => {
            let displayName = stat;
            let displayValue = predictions[stat];

        if (stat === 'FG_PCT') {
            displayName = 'FG%';
            displayValue = `${(displayValue * 100).toFixed(1)}%`;
        } else if (stat === 'FT_PCT') {
            displayName = 'FT%';
            displayValue = `${(displayValue * 100).toFixed(1)}%`;
        } else {
            displayValue = displayValue.toFixed(1);
        }

        return { displayName, displayValue };
    });

    // split into two columns (max 5 items per column)
    const midpoint = Math.ceil(formattedPredictions.length / 2);
    const firstColumn = formattedPredictions.slice(0, midpoint);
    const secondColumn = formattedPredictions.slice(midpoint);

    return (
        <div style={{ display: 'flex', gap: '24px', alignItems: 'center', marginTop: '40px' }}>
            {/* player photo and name */}
            <div style={{ textAlign: 'center' }}>
                <img
                    src={headshot_url}
                    alt={player_name}
                    style={{ width: '200px', borderRadius: '12px' }}
                />
                <h2 style={{ margin: '10px 0 4px' }}>{player_name}</h2>
            </div>

            {/* stat line with two columns */}
            <div>
                <h3 style={{ marginBottom: '12px' }}>Predicted Stat Line for Next Game</h3>
                <div style={{ display: 'flex', gap: '40px' }}>
                    <ul style={{ listStyle: 'none', padding: 0 }}>
                        {firstColumn.map(({ displayName, displayValue }) => (
                            <li key={displayName} style={{ marginBottom: '6px' }}>
                                <strong>{displayName}:</strong> {displayValue}
                            </li>
                        ))}
                    </ul>
                    <ul style={{ listStyle: 'none', padding: 0 }}>
                        {secondColumn.map(({ displayName, displayValue }) => (
                            <li key={displayName} style={{ marginBottom: '6px' }}>
                                <strong>{displayName}:</strong> {displayValue}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default PredictionDisplay;