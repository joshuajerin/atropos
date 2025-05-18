const express = require('express');
const cors = require('cors');
const path = require('path');
const app = express();
const PORT = 3000;

// Enable CORS for all routes
app.use(cors());

// Parse JSON request bodies
app.use(express.json());

// Serve static files from the current directory
app.use(express.static(__dirname));

// Global variable to store the latest cube state
let latestCubeState = null;

// API endpoint to get the current cube matrix
app.get('/api/cube-state', (req, res) => {
    if (latestCubeState) {
        res.json(latestCubeState);
    } else {
        res.status(404).json({ error: 'Cube state not available yet. Please generate it first using the UI.' });
    }
});

// API endpoint to update the cube state
app.post('/api/cube-state', (req, res) => {
    try {
        latestCubeState = req.body;
        console.log('Cube state updated via POST request');
        res.status(200).json({ message: 'Cube state updated successfully' });
    } catch (error) {
        console.error('Error updating cube state:', error);
        res.status(500).json({ error: 'Failed to update cube state' });
    }
});

// API endpoint to perform a move on the cube
app.post('/api/move', (req, res) => {
    try {
        const move = req.body.move;
        if (!move) {
            return res.status(400).json({ error: 'Move parameter is required' });
        }
        
        // Emit an event to notify the client about the move
        console.log(`Received move command: ${move}`);
        
        // Store the move in a global variable that the client can poll for
        global.pendingMoves = global.pendingMoves || [];
        global.pendingMoves.push(move);
        
        res.status(200).json({ message: `Move '${move}' queued successfully` });
    } catch (error) {
        console.error('Error processing move:', error);
        res.status(500).json({ error: 'Failed to process move' });
    }
});

// API endpoint to get pending moves (for polling)
app.get('/api/pending-moves', (req, res) => {
    const moves = global.pendingMoves || [];
    global.pendingMoves = [];  // Clear the moves after retrieving them
    res.json({ moves });
});

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log(`API endpoint available at http://localhost:${PORT}/api/cube-state`);
});

// No need to export updateCubeState anymore as we're using the POST endpoint
