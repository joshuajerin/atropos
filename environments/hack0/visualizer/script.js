// Rubik's Cube Visualizer

// Constants
const CUBE_SIZE = 3; // 3x3x3 standard Rubik's cube
const CUBIE_SIZE = 1;
const CUBIE_GAP = 0.1;
const TOTAL_SIZE = CUBE_SIZE * (CUBIE_SIZE + CUBIE_GAP);

// Colors for the six faces
const COLORS = {
    UP: 0xffffff,    // White
    DOWN: 0xffff00,  // Yellow
    FRONT: 0x00ff00, // Green
    BACK: 0x0000ff,  // Blue
    RIGHT: 0xff0000, // Red
    LEFT: 0xff8800   // Orange
};

// Define the face colors for visualization (letter-based)
const UP_COLOR = 'W';     // White
const DOWN_COLOR = 'Y';   // Yellow
const RIGHT_COLOR = 'R';  // Red
const LEFT_COLOR = 'O';   // Orange
const FRONT_COLOR = 'G';  // Green
const BACK_COLOR = 'B';   // Blue

// Cube state representation
let cubeState = createInitialCubeState();

// Three.js variables
let scene, camera, renderer, controls;
let cubeGroup = new THREE.Group();
let cubies = [];

// Initialize the 3D scene
function init() {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Create camera
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(5, 5, 7);
    camera.lookAt(0, 0, 0);
    
    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(document.getElementById('cube-container').clientWidth, 
                     document.getElementById('cube-container').clientHeight);
    document.getElementById('cube-container').appendChild(renderer.domElement);
    
    // Add orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    
    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(10, 20, 15);
    scene.add(directionalLight);
    
    // Create the cube
    createCube();
    
    // Add the cube to the scene
    scene.add(cubeGroup);
    
    // Start animation loop
    animate();
    
    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

// Create the initial cube state
function createInitialCubeState() {
    const state = {};
    
    // Initialize each face with its letter-based color
    state.up = Array(CUBE_SIZE).fill().map(() => Array(CUBE_SIZE).fill(UP_COLOR));
    state.down = Array(CUBE_SIZE).fill().map(() => Array(CUBE_SIZE).fill(DOWN_COLOR));
    state.front = Array(CUBE_SIZE).fill().map(() => Array(CUBE_SIZE).fill(FRONT_COLOR));
    state.back = Array(CUBE_SIZE).fill().map(() => Array(CUBE_SIZE).fill(BACK_COLOR));
    state.right = Array(CUBE_SIZE).fill().map(() => Array(CUBE_SIZE).fill(RIGHT_COLOR));
    state.left = Array(CUBE_SIZE).fill().map(() => Array(CUBE_SIZE).fill(LEFT_COLOR));
    
    return state;
}

// Create the 3D cube model
function createCube() {
    cubies = [];
    cubeGroup.clear();
    
    const offset = (CUBE_SIZE - 1) / 2;
    
    // Create each cubie
    for (let x = 0; x < CUBE_SIZE; x++) {
        for (let y = 0; y < CUBE_SIZE; y++) {
            for (let z = 0; z < CUBE_SIZE; z++) {
                // Skip internal cubies (not visible)
                if (x > 0 && x < CUBE_SIZE - 1 && 
                    y > 0 && y < CUBE_SIZE - 1 && 
                    z > 0 && z < CUBE_SIZE - 1) {
                    continue;
                }
                
                const cubie = createCubie(x, y, z);
                
                // Position the cubie
                cubie.position.set(
                    (x - offset) * (CUBIE_SIZE + CUBIE_GAP),
                    (y - offset) * (CUBIE_SIZE + CUBIE_GAP),
                    (z - offset) * (CUBIE_SIZE + CUBIE_GAP)
                );
                
                cubies.push({
                    mesh: cubie,
                    position: { x, y, z }
                });
                
                cubeGroup.add(cubie);
            }
        }
    }
}

// Create a single cubie with colored faces
function createCubie(x, y, z) {
    const cubie = new THREE.Group();
    
    // Create geometry and materials
    const geometry = new THREE.BoxGeometry(CUBIE_SIZE, CUBIE_SIZE, CUBIE_SIZE);
    const materials = Array(6).fill().map(() => new THREE.MeshLambertMaterial({ color: 0x333333 }));
    
    // Get colors from the cube state
    // Set colors for visible faces based on position and cube state
    if (y === CUBE_SIZE - 1) {
        const faceColor = getFaceColor(cubeState.up[z][x]);
        materials[2].color.setHex(faceColor);      // Up face
    }
    if (y === 0) {
        const faceColor = getFaceColor(cubeState.down[CUBE_SIZE - 1 - z][x]);
        materials[3].color.setHex(faceColor);      // Down face
    }
    if (z === CUBE_SIZE - 1) {
        const faceColor = getFaceColor(cubeState.front[CUBE_SIZE - 1 - y][x]);
        materials[4].color.setHex(faceColor);      // Front face
    }
    if (z === 0) {
        const faceColor = getFaceColor(cubeState.back[CUBE_SIZE - 1 - y][CUBE_SIZE - 1 - x]);
        materials[5].color.setHex(faceColor);      // Back face
    }
    if (x === CUBE_SIZE - 1) {
        const faceColor = getFaceColor(cubeState.right[CUBE_SIZE - 1 - y][CUBE_SIZE - 1 - z]);
        materials[0].color.setHex(faceColor);      // Right face
    }
    if (x === 0) {
        const faceColor = getFaceColor(cubeState.left[CUBE_SIZE - 1 - y][z]);
        materials[1].color.setHex(faceColor);      // Left face
    }
    
    const mesh = new THREE.Mesh(geometry, materials);
    cubie.add(mesh);
    
    return cubie;
}

// Helper function to get the color based on the face value
function getFaceColor(faceValue) {
    switch(faceValue) {
        case UP_COLOR: return COLORS.UP;
        case DOWN_COLOR: return COLORS.DOWN;
        case FRONT_COLOR: return COLORS.FRONT;
        case BACK_COLOR: return COLORS.BACK;
        case RIGHT_COLOR: return COLORS.RIGHT;
        case LEFT_COLOR: return COLORS.LEFT;
        default: return 0x333333; // Default gray
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    
    // Check for pending moves from the API (every few seconds)
    if (!window.lastPollTime || Date.now() - window.lastPollTime > 1000) {  // Poll every 1 second
        window.lastPollTime = Date.now();
        checkPendingMoves();
    }
}

// Function to check for pending moves from the API
function checkPendingMoves() {
    fetch('/api/pending-moves')
        .then(response => response.json())
        .then(data => {
            if (data.moves && data.moves.length > 0) {
                console.log('Received moves from API:', data.moves);
                // Process each move
                data.moves.forEach(move => {
                    // Display the move in the UI
                    document.getElementById('current-move').textContent = `Move: ${move}`;
                    document.getElementById('move-description').textContent = 'Move received via API';
                    
                    // Perform the move
                    performMove(move);
                    updateCubeDisplay();
                    
                    // Log for debugging
                    console.log('Executed move from API:', move);
                });
            }
        })
        .catch(error => {
            console.error('Error checking for pending moves:', error);
        });
}

// Handle window resize
function onWindowResize() {
    const container = document.getElementById('cube-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// Randomize the cube
function randomizeCube() {
    const moves = ['U', 'D', 'F', 'B', 'R', 'L', "U'", "D'", "F'", "B'", "R'", "L'"];
    const numMoves = 20; // Number of random moves to perform
    
    console.log('Randomizing cube with', numMoves, 'moves');
    
    for (let i = 0; i < numMoves; i++) {
        const randomMove = moves[Math.floor(Math.random() * moves.length)];
        console.log('Move', i + 1, ':', randomMove);
        performMove(randomMove);
    }
    
    // Make sure to update the 3D display after all moves
    updateCubeDisplay();
}

// Perform a single move on the cube
function performMove(move) {
    // Store the current state
    const prevState = JSON.parse(JSON.stringify(cubeState));
    
    // Determine the face and direction
    const face = move.charAt(0).toLowerCase();
    // Check if the move has a prime symbol (') which means counterclockwise
    const clockwise = !move.includes("'");
    // Check if the move is a double move (180° rotation)
    const isDoubleMove = move.includes("2");
    
    // Perform the appropriate rotation
    switch (face) {
        case 'u': 
            rotateUpFace(clockwise);
            // For double moves, perform the same rotation twice
            if (isDoubleMove) rotateUpFace(clockwise);
            break;
        case 'd': 
            rotateDownFace(clockwise);
            if (isDoubleMove) rotateDownFace(clockwise);
            break;
        case 'f': 
            rotateFrontFace(clockwise);
            if (isDoubleMove) rotateFrontFace(clockwise);
            break;
        case 'b': 
            rotateBackFace(clockwise);
            if (isDoubleMove) rotateBackFace(clockwise);
            break;
        case 'r': 
            rotateRightFace(clockwise);
            if (isDoubleMove) rotateRightFace(clockwise);
            break;
        case 'l': 
            rotateLeftFace(clockwise);
            if (isDoubleMove) rotateLeftFace(clockwise);
            break;
    }
}

// Rotate functions for each face
function rotateUpFace(clockwise) {
    // Rotate the up face itself
    rotateFace(cubeState.up, clockwise);
    
    // Rotate the adjacent edges
    const temp = [...cubeState.front[CUBE_SIZE - 1]];
    
    if (clockwise) {
        // front -> right -> back -> left -> front
        cubeState.front[CUBE_SIZE - 1] = [...cubeState.left[CUBE_SIZE - 1]];
        cubeState.left[CUBE_SIZE - 1] = [...cubeState.back[CUBE_SIZE - 1]];
        cubeState.back[CUBE_SIZE - 1] = [...cubeState.right[CUBE_SIZE - 1]];
        cubeState.right[CUBE_SIZE - 1] = temp;
    } else {
        // front -> left -> back -> right -> front
        cubeState.front[CUBE_SIZE - 1] = [...cubeState.right[CUBE_SIZE - 1]];
        cubeState.right[CUBE_SIZE - 1] = [...cubeState.back[CUBE_SIZE - 1]];
        cubeState.back[CUBE_SIZE - 1] = [...cubeState.left[CUBE_SIZE - 1]];
        cubeState.left[CUBE_SIZE - 1] = temp;
    }
}

function rotateDownFace(clockwise) {
    // Rotate the down face itself
    rotateFace(cubeState.down, clockwise);
    
    // Rotate the adjacent edges
    const temp = [...cubeState.front[0]];
    
    if (clockwise) {
        // front -> left -> back -> right -> front
        cubeState.front[0] = [...cubeState.right[0]];
        cubeState.right[0] = [...cubeState.back[0]];
        cubeState.back[0] = [...cubeState.left[0]];
        cubeState.left[0] = temp;
    } else {
        // front -> right -> back -> left -> front
        cubeState.front[0] = [...cubeState.left[0]];
        cubeState.left[0] = [...cubeState.back[0]];
        cubeState.back[0] = [...cubeState.right[0]];
        cubeState.right[0] = temp;
    }
}

function rotateFrontFace(clockwise) {
    // Rotate the front face itself
    rotateFace(cubeState.front, clockwise);
    
    // Get the rows and columns that need to be rotated
    const tempUp = cubeState.up[0].map((_, i) => cubeState.up[CUBE_SIZE - 1 - i][0]);
    
    if (clockwise) {
        // up -> right -> down -> left -> up
        for (let i = 0; i < CUBE_SIZE; i++) {
            const rightVal = cubeState.right[i][0];
            const downVal = cubeState.down[0][CUBE_SIZE - 1 - i];
            const leftVal = cubeState.left[CUBE_SIZE - 1 - i][CUBE_SIZE - 1];
            
            cubeState.right[i][0] = tempUp[i];
            cubeState.down[0][CUBE_SIZE - 1 - i] = rightVal;
            cubeState.left[CUBE_SIZE - 1 - i][CUBE_SIZE - 1] = downVal;
            cubeState.up[CUBE_SIZE - 1 - i][0] = leftVal;
        }
    } else {
        // up -> left -> down -> right -> up
        for (let i = 0; i < CUBE_SIZE; i++) {
            const leftVal = cubeState.left[CUBE_SIZE - 1 - i][CUBE_SIZE - 1];
            const downVal = cubeState.down[0][CUBE_SIZE - 1 - i];
            const rightVal = cubeState.right[i][0];
            
            cubeState.left[CUBE_SIZE - 1 - i][CUBE_SIZE - 1] = tempUp[i];
            cubeState.down[0][CUBE_SIZE - 1 - i] = leftVal;
            cubeState.right[i][0] = downVal;
            cubeState.up[CUBE_SIZE - 1 - i][0] = rightVal;
        }
    }
}

function rotateBackFace(clockwise) {
    // Rotate the back face itself
    rotateFace(cubeState.back, clockwise);
    
    // Get the rows and columns that need to be rotated
    const tempUp = cubeState.up[0].map((_, i) => cubeState.up[i][CUBE_SIZE - 1]);
    
    if (clockwise) {
        // up -> left -> down -> right -> up
        for (let i = 0; i < CUBE_SIZE; i++) {
            const leftVal = cubeState.left[i][0];
            const downVal = cubeState.down[CUBE_SIZE - 1][i];
            const rightVal = cubeState.right[CUBE_SIZE - 1 - i][CUBE_SIZE - 1];
            
            cubeState.left[i][0] = tempUp[i];
            cubeState.down[CUBE_SIZE - 1][i] = leftVal;
            cubeState.right[CUBE_SIZE - 1 - i][CUBE_SIZE - 1] = downVal;
            cubeState.up[i][CUBE_SIZE - 1] = rightVal;
        }
    } else {
        // up -> right -> down -> left -> up
        for (let i = 0; i < CUBE_SIZE; i++) {
            const rightVal = cubeState.right[CUBE_SIZE - 1 - i][CUBE_SIZE - 1];
            const downVal = cubeState.down[CUBE_SIZE - 1][i];
            const leftVal = cubeState.left[i][0];
            
            cubeState.right[CUBE_SIZE - 1 - i][CUBE_SIZE - 1] = tempUp[i];
            cubeState.down[CUBE_SIZE - 1][i] = rightVal;
            cubeState.left[i][0] = downVal;
            cubeState.up[i][CUBE_SIZE - 1] = leftVal;
        }
    }
}

function rotateRightFace(clockwise) {
    // Rotate the right face itself
    rotateFace(cubeState.right, clockwise);
    
    // Get the columns that need to be rotated
    const tempUp = cubeState.up.map(row => row[CUBE_SIZE - 1]);
    
    if (clockwise) {
        // up -> back -> down -> front -> up
        for (let i = 0; i < CUBE_SIZE; i++) {
            const backVal = cubeState.back[CUBE_SIZE - 1 - i][0];
            const downVal = cubeState.down[i][CUBE_SIZE - 1];
            const frontVal = cubeState.front[i][CUBE_SIZE - 1];
            
            cubeState.back[CUBE_SIZE - 1 - i][0] = tempUp[i];
            cubeState.down[i][CUBE_SIZE - 1] = backVal;
            cubeState.front[i][CUBE_SIZE - 1] = downVal;
            cubeState.up[i][CUBE_SIZE - 1] = frontVal;
        }
    } else {
        // up -> front -> down -> back -> up
        for (let i = 0; i < CUBE_SIZE; i++) {
            const frontVal = cubeState.front[i][CUBE_SIZE - 1];
            const downVal = cubeState.down[i][CUBE_SIZE - 1];
            const backVal = cubeState.back[CUBE_SIZE - 1 - i][0];
            
            cubeState.front[i][CUBE_SIZE - 1] = tempUp[i];
            cubeState.down[i][CUBE_SIZE - 1] = frontVal;
            cubeState.back[CUBE_SIZE - 1 - i][0] = downVal;
            cubeState.up[i][CUBE_SIZE - 1] = backVal;
        }
    }
}

function rotateLeftFace(clockwise) {
    // Rotate the left face itself
    rotateFace(cubeState.left, clockwise);
    
    // Get the columns that need to be rotated
    const tempUp = cubeState.up.map(row => row[0]);
    
    if (clockwise) {
        // up -> front -> down -> back -> up
        for (let i = 0; i < CUBE_SIZE; i++) {
            const frontVal = cubeState.front[i][0];
            const downVal = cubeState.down[i][0];
            const backVal = cubeState.back[CUBE_SIZE - 1 - i][CUBE_SIZE - 1];
            
            cubeState.front[i][0] = tempUp[i];
            cubeState.down[i][0] = frontVal;
            cubeState.back[CUBE_SIZE - 1 - i][CUBE_SIZE - 1] = downVal;
            cubeState.up[i][0] = backVal;
        }
    } else {
        // up -> back -> down -> front -> up
        for (let i = 0; i < CUBE_SIZE; i++) {
            const backVal = cubeState.back[CUBE_SIZE - 1 - i][CUBE_SIZE - 1];
            const downVal = cubeState.down[i][0];
            const frontVal = cubeState.front[i][0];
            
            cubeState.back[CUBE_SIZE - 1 - i][CUBE_SIZE - 1] = tempUp[i];
            cubeState.down[i][0] = backVal;
            cubeState.front[i][0] = downVal;
            cubeState.up[i][0] = frontVal;
        }
    }
}

// Helper function to rotate a face (2D array) clockwise or counterclockwise
function rotateFace(face, clockwise) {
    const size = face.length;
    const newFace = Array(size).fill().map(() => Array(size).fill(null));
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            if (clockwise) {
                newFace[j][size - 1 - i] = face[i][j];
            } else {
                newFace[size - 1 - j][i] = face[i][j];
            }
        }
    }
    
    // Copy the rotated face back to the original
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            face[i][j] = newFace[i][j];
        }
    }
}

// Update the 3D cube display based on the current state
function updateCubeDisplay() {
    // Update the 3D representation based on the current state
    createCube();
    
    // Automatically update the server with the current cube state
    updateServerCubeState();
    
    // Log the updated state for debugging
    console.log('Cube state updated:', cubeState);
}

// Function to update the server with the current cube state
function updateServerCubeState() {
    // Create a JSON object with the cube state
    const cubeStateJSON = {
        timestamp: new Date().toISOString(),
        cubeSize: CUBE_SIZE,
        faces: {
            U: cubeState.up,
            D: cubeState.down,
            F: cubeState.front,
            B: cubeState.back,
            R: cubeState.right,
            L: cubeState.left
        }
    };
    
    // Send the cube state to the server API
    fetch('/api/cube-state', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(cubeStateJSON)
    })
    .then(response => {
        if (response.ok) {
            console.log('Cube state automatically updated on server');
        }
    })
    .catch(error => {
        console.error('Error updating server cube state:', error);
    });
}

// Reset the cube to its initial solved state
function resetCube() {
    cubeState = createInitialCubeState();
    updateCubeDisplay();
}

// Export the cube state as JSON
function exportCubeStateJSON() {
    // Create a JSON object with the cube state
    const cubeStateJSON = {
        timestamp: new Date().toISOString(),
        cubeSize: CUBE_SIZE,
        faces: {
            U: cubeState.up,
            D: cubeState.down,
            F: cubeState.front,
            B: cubeState.back,
            R: cubeState.right,
            L: cubeState.left
        }
    };
    
    // Convert to JSON string
    const jsonString = JSON.stringify(cubeStateJSON, null, 2);
    
    // Create a blob with the JSON data
    const blob = new Blob([jsonString], { type: 'application/json' });
    
    // Create a URL for the blob
    const url = URL.createObjectURL(blob);
    
    // Create a temporary link element
    const link = document.createElement('a');
    link.href = url;
    link.download = 'rubiks_cube_state.json';
    
    // Append the link to the body
    document.body.appendChild(link);
    
    // Trigger the download
    link.click();
    
    // Clean up
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    // Also send the cube state to the server API
    fetch('/api/cube-state', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: jsonString
    })
    .then(response => {
        if (response.ok) {
            console.log('Cube state sent to server successfully');
        } else {
            console.error('Failed to send cube state to server');
        }
    })
    .catch(error => {
        console.error('Error sending cube state to server:', error);
    });
    
    console.log('Cube state exported as JSON');
    return jsonString;
}

// Format a face matrix as a string (kept for potential future use)
function formatFaceMatrix(face) {
    return face.map(row => row.join(' ')).join('\n');
}

// Solve the cube with a predefined sequence of moves
let isSolving = false;
let moveQueue = [];
let moveDescriptions = {};

function solveCube() {
    if (isSolving) return; // Don't start a new solve sequence if one is already in progress
    
    // Clear any existing moves
    moveQueue = [];
    moveDescriptions = {};
    
    // Define a small set of moves with descriptions
    const moves = [
        { move: "R2", description: "Rotating the right face clockwise (aligns the right edge pieces)" },
        { move: "U'", description: "Rotating the top face counter-clockwise (positions the top corner)" },
        { move: "F", description: "Rotating the front face clockwise (brings the front edge into position)" },
        { move: "D", description: "Rotating the bottom face clockwise (aligns the bottom layer)" },
        { move: "L'", description: "Rotating the left face counter-clockwise (completes the left side)" }
    ];
    
    // Reset the cube first to ensure we're starting from a solved state
    resetCube();
    
    // Apply a simple scramble (just 3 moves) to show the solving process
    // const scrambleMoves = ["R", "U", "F"];
    // for (let i = 0; i < scrambleMoves.length; i++) {
    //     const move = scrambleMoves[i];
    //     moveQueue.push(move);
    //     moveDescriptions[move] = "Scrambling the cube";
    // }
    
    // Add a small delay before starting the solution
    setTimeout(() => {
        // Add solution moves to the queue
        for (const moveInfo of moves) {
            moveQueue.push(moveInfo.move);
            moveDescriptions[moveInfo.move] = moveInfo.description;
        }
        
        // Start processing the move queue
        processNextMove();
    }, 1000);
}

// Process the next move in the queue with animation
function processNextMove() {
    // Clear the status display if no moves left
    if (moveQueue.length === 0) {
        isSolving = false;
        document.getElementById('current-move').textContent = '';
        document.getElementById('move-description').textContent = 'Cube solving completed!';
        setTimeout(() => {
            document.getElementById('move-description').textContent = '';
        }, 3000);
        return;
    }
    
    isSolving = true;
    const nextMove = moveQueue.shift();
    
    // Display the current move in the UI
    document.getElementById('current-move').textContent = `Move: ${nextMove}`;
    document.getElementById('move-description').textContent = moveDescriptions[nextMove] || '';
    
    // Log for debugging
    console.log('Executing move:', nextMove, moveDescriptions[nextMove]);
    
    // Perform the move
    performMove(nextMove);
    updateCubeDisplay();
    
    // Wait a bit before the next move to create an animation effect
    setTimeout(processNextMove, 1000); // Increased delay to 1 second for better visibility
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the 3D scene
    init();
    
    // Add event listeners for buttons
    document.getElementById('randomize-btn').addEventListener('click', () => {
        randomizeCube();
    });
    
    document.getElementById('reset-btn').addEventListener('click', () => {
        resetCube();
    });
    
    document.getElementById('solve-btn').addEventListener('click', () => {
        solveCube();
    });
    
    document.getElementById('get-matrix-btn').addEventListener('click', () => {
        exportCubeStateJSON();
    });
});
