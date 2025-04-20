document.addEventListener('DOMContentLoaded', function() {
    // Neural Network Configuration
    const config = {
        layers: [2, 3, 1], // Default network architecture
        learningRate: 0.1,
        activation: x => 1 / (1 + Math.exp(-x)), // Sigmoid
        derivative: x => {
            const sig = 1 / (1 + Math.exp(-x));
            return sig * (1 - sig);
        }
    };

    // Neural Network State
    let network = {
        weights: [],
        biases: [],
        activations: []
    };

    // UI Elements
    const networkContainer = document.getElementById('network');
    const addLayerBtn = document.getElementById('add-layer');
    const removeLayerBtn = document.getElementById('remove-layer');
    const layerCountDisplay = document.getElementById('layer-count');
    const learningRateSlider = document.getElementById('learning-rate');
    const lrValueDisplay = document.getElementById('lr-value');
    const trainBtn = document.getElementById('train-btn');
    const resetBtn = document.getElementById('reset-btn');
    const hackModeBtn = document.getElementById('hack-mode-btn');
    const hackPanel = document.getElementById('hack-panel');
    const weightSlidersContainer = document.getElementById('weight-sliders');
    const biasSlidersContainer = document.getElementById('bias-sliders');

    // Digit Recognizer Elements
    const digitCanvas = document.getElementById('digit-canvas');
    const clearCanvasBtn = document.getElementById('clear-canvas');
    const predictDigitBtn = document.getElementById('predict-digit');
    const predictionOutput = document.getElementById('prediction-output');
    const confidenceBar = document.getElementById('confidence-bar');

    // Gradient Descent Elements
    const gradientCanvas = document.getElementById('gradient-canvas');
    const startGradientBtn = document.getElementById('start-gradient');
    const resetGradientBtn = document.getElementById('reset-gradient');

    // Initialize the application
    initNetwork();
    renderNetwork();
    setupDigitCanvas();
    setupGradientDescent();

    // Event Listeners
    addLayerBtn.addEventListener('click', addLayer);
    removeLayerBtn.addEventListener('click', removeLayer);
    learningRateSlider.addEventListener('input', updateLearningRate);
    trainBtn.addEventListener('click', trainNetwork);
    resetBtn.addEventListener('click', resetNetwork);
    hackModeBtn.addEventListener('click', toggleHackMode);
    clearCanvasBtn.addEventListener('click', clearDigitCanvas);
    predictDigitBtn.addEventListener('click', predictDigit);
    startGradientBtn.addEventListener('click', startGradientDescent);
    resetGradientBtn.addEventListener('click', resetGradientDescent);

    // Initialize Neural Network
    function initNetwork() {
        // Initialize weights and biases
        network.weights = [];
        network.biases = [];
        
        for (let i = 0; i < config.layers.length - 1; i++) {
            const rows = config.layers[i + 1];
            const cols = config.layers[i];
            
            // Initialize weights with random values between -1 and 1
            const weights = new Array(rows);
            for (let j = 0; j < rows; j++) {
                weights[j] = new Array(cols);
                for (let k = 0; k < cols; k++) {
                    weights[j][k] = Math.random() * 2 - 1;
                }
            }
            network.weights.push(weights);
            
            // Initialize biases with random values between -1 and 1
            const biases = new Array(rows);
            for (let j = 0; j < rows; j++) {
                biases[j] = Math.random() * 2 - 1;
            }
            network.biases.push(biases);
        }
    }

    // Render the neural network visualization
    function renderNetwork() {
        networkContainer.innerHTML = '';
        
        // Create layers
        for (let i = 0; i < config.layers.length; i++) {
            const layer = document.createElement('div');
            layer.className = 'layer';
            layer.dataset.layerIndex = i;
            
            // Create neurons
            for (let j = 0; j < config.layers[i]; j++) {
                const neuron = document.createElement('div');
                neuron.className = 'neuron';
                neuron.dataset.layerIndex = i;
                neuron.dataset.neuronIndex = j;
                neuron.textContent = j + 1;
                layer.appendChild(neuron);
            }
            
            networkContainer.appendChild(layer);
        }
        
        // Create connections between layers
        for (let i = 0; i < config.layers.length - 1; i++) {
            const currentLayer = networkContainer.children[i];
            const nextLayer = networkContainer.children[i + 1];
            
            for (let j = 0; j < currentLayer.children.length; j++) {
                for (let k = 0; k < nextLayer.children.length; k++) {
                    const connection = document.createElement('div');
                    connection.className = 'connection';
                    
                    const startNeuron = currentLayer.children[j];
                    const endNeuron = nextLayer.children[k];
                    
                    const startRect = startNeuron.getBoundingClientRect();
                    const endRect = endNeuron.getBoundingClientRect();
                    
                    const networkRect = networkContainer.getBoundingClientRect();
                    
                    const startX = startRect.left + startRect.width / 2 - networkRect.left;
                    const startY = startRect.top + startRect.height / 2 - networkRect.top;
                    const endX = endRect.left + endRect.width / 2 - networkRect.left;
                    const endY = endRect.top + endRect.height / 2 - networkRect.top;
                    
                    const length = Math.sqrt(Math.pow(endX - startX, 2) + Math.pow(endY - startY, 2));
                    const angle = Math.atan2(endY - startY, endX - startX);
                    
                    connection.style.width = `${length}px`;
                    connection.style.left = `${startX}px`;
                    connection.style.top = `${startY}px`;
                    connection.style.transform = `rotate(${angle}rad)`;
                    connection.style.opacity = Math.abs(network.weights[i][k][j] / 2 + 0.5);
                    
                    networkContainer.appendChild(connection);
                }
            }
        }
        
        // Update layer count display
        layerCountDisplay.textContent = config.layers.length;
    }

    // Add a layer to the network
    function addLayer() {
        if (config.layers.length < 6) {
            // Add a new layer with 3 neurons before the output layer
            const insertAt = config.layers.length - 1;
            config.layers.splice(insertAt, 0, 3);
            initNetwork();
            renderNetwork();
        } else {
            alert('Maximum number of layers reached (6)');
        }
    }

    // Remove a layer from the network
    function removeLayer() {
        if (config.layers.length > 2) {
            // Remove the last hidden layer (keep at least input and output layers)
            const removeAt = config.layers.length - 2;
            config.layers.splice(removeAt, 1);
            initNetwork();
            renderNetwork();
        } else {
            alert('Minimum number of layers reached (2)');
        }
    }

    // Update learning rate from slider
    function updateLearningRate() {
        config.learningRate = parseFloat(learningRateSlider.value);
        lrValueDisplay.textContent = config.learningRate.toFixed(2);
    }

    // Train the network on XOR problem
    function trainNetwork() {
        // XOR training data
        const trainingData = [
            { input: [0, 0], output: [0] },
            { input: [0, 1], output: [1] },
            { input: [1, 0], output: [1] },
            { input: [1, 1], output: [0] }
        ];
        
        // Train for 1000 epochs
        for (let epoch = 0; epoch < 1000; epoch++) {
            for (const data of trainingData) {
                // Forward pass
                const activations = forwardPass(data.input);
                
                // Backward pass
                backpropagate(data.input, data.output, activations);
            }
        }
        
        // Visualize the training
        animateDataFlow(trainingData);
    }

    // Perform forward pass
    function forwardPass(input) {
        const activations = [input];
        let currentActivation = input;
        
        for (let i = 0; i < network.weights.length; i++) {
            const layerWeights = network.weights[i];
            const layerBiases = network.biases[i];
            
            const nextActivation = new Array(layerWeights.length);
            
            for (let j = 0; j < layerWeights.length; j++) {
                let weightedSum = layerBiases[j];
                
                for (let k = 0; k < layerWeights[j].length; k++) {
                    weightedSum += layerWeights[j][k] * currentActivation[k];
                }
                
                nextActivation[j] = config.activation(weightedSum);
            }
            
            currentActivation = nextActivation;
            activations.push(currentActivation);
        }
        
        return activations;
    }

    // Perform backpropagation
    function backpropagate(input, target, activations) {
        // Calculate output errors
        const errors = [];
        const outputErrors = new Array(activations[activations.length - 1].length);
        
        for (let i = 0; i < outputErrors.length; i++) {
            const output = activations[activations.length - 1][i];
            outputErrors[i] = (output - target[i]) * config.derivative(output);
        }
        
        errors.unshift(outputErrors);
        
        // Calculate hidden layer errors
        for (let i = network.weights.length - 1; i > 0; i--) {
            const currentErrors = new Array(network.weights[i - 1][0].length);
            
            for (let j = 0; j < currentErrors.length; j++) {
                let errorSum = 0;
                
                for (let k = 0; k < network.weights[i].length; k++) {
                    errorSum += network.weights[i][k][j] * errors[0][k];
                }
                
                currentErrors[j] = errorSum * config.derivative(activations[i][j]);
            }
            
            errors.unshift(currentErrors);
        }
        
        // Update weights and biases
        for (let i = 0; i < network.weights.length; i++) {
            for (let j = 0; j < network.weights[i].length; j++) {
                // Update bias
                network.biases[i][j] -= config.learningRate * errors[i][j];
                
                // Update weights
                for (let k = 0; k < network.weights[i][j].length; k++) {
                    network.weights[i][j][k] -= config.learningRate * errors[i][j] * activations[i][k];
                }
            }
        }
    }

    // Animate data flow through the network
    function animateDataFlow(trainingData) {
        const dataFlowContainer = document.getElementById('data-flow');
        dataFlowContainer.innerHTML = '';
        
        // Get all neuron positions
        const neuronPositions = [];
        const layers = document.querySelectorAll('.layer');
        
        layers.forEach((layer, layerIndex) => {
            const neurons = layer.querySelectorAll('.neuron');
            const layerPositions = [];
            
            neurons.forEach((neuron, neuronIndex) => {
                const rect = neuron.getBoundingClientRect();
                const containerRect = networkContainer.getBoundingClientRect();
                
                layerPositions.push({
                    x: rect.left + rect.width / 2 - containerRect.left,
                    y: rect.top + rect.height / 2 - containerRect.top
                });
            });
            
            neuronPositions.push(layerPositions);
        });
        
        // Animate each training example
        trainingData.forEach((data, index) => {
            setTimeout(() => {
                // Highlight input neurons
                const inputLayer = layers[0];
                const inputNeurons = inputLayer.querySelectorAll('.neuron');
                
                inputNeurons.forEach((neuron, i) => {
                    neuron.classList.add('active');
                    neuron.style.backgroundColor = data.input[i] === 1 ? '#4CAF50' : '#F44336';
                });
                
                // Create data points that flow through the network
                const activations = forwardPass(data.input);
                
                activations.forEach((layerActivations, layerIndex) => {
                    layerActivations.forEach((activation, neuronIndex) => {
                        setTimeout(() => {
                            // Create a data point
                            const dataPoint = document.createElement('div');
                            dataPoint.className = 'data-flow';
                            dataPoint.style.left = `${neuronPositions[layerIndex][neuronIndex].x}px`;
                            dataPoint.style.top = `${neuronPositions[layerIndex][neuronIndex].y}px`;
                            dataPoint.style.backgroundColor = activation > 0.5 ? '#4CAF50' : '#F44336';
                            dataPoint.style.opacity = activation;
                            dataPoint.style.transform = `scale(${activation * 2})`;
                            
                            dataFlowContainer.appendChild(dataPoint);
                            
                            // Highlight neuron
                            if (layerIndex < layers.length - 1) {
                                const neuron = layers[layerIndex + 1].children[neuronIndex];
                                neuron.classList.add('active');
                                neuron.style.backgroundColor = activation > 0.5 ? '#4CAF50' : '#F44336';
                            }
                            
                            // Remove data point after animation
                            setTimeout(() => {
                                dataPoint.remove();
                                if (layerIndex === layers.length - 1 && neuronIndex === layerActivations.length - 1) {
                                    // Reset neuron highlights
                                    const allNeurons = document.querySelectorAll('.neuron');
                                    allNeurons.forEach(n => {
                                        n.classList.remove('active');
                                        n.style.backgroundColor = '';
                                    });
                                }
                            }, 1000);
                        }, layerIndex * 500 + neuronIndex * 100);
                    });
                });
            }, index * 3000);
        });
    }

    // Reset the network to initial state
    function resetNetwork() {
        initNetwork();
        renderNetwork();
    }

    // Toggle hack mode to adjust weights and biases manually
    function toggleHackMode() {
        hackPanel.style.display = hackPanel.style.display === 'none' ? 'block' : 'none';
        
        if (hackPanel.style.display === 'block') {
            // Create weight sliders
            weightSlidersContainer.innerHTML = '';
            
            for (let i = 0; i < network.weights.length; i++) {
                const layerDiv = document.createElement('div');
                layerDiv.innerHTML = `<h4>Layer ${i} to ${i + 1}</h4>`;
                weightSlidersContainer.appendChild(layerDiv);
                
                for (let j = 0; j < network.weights[i].length; j++) {
                    for (let k = 0; k < network.weights[i][j].length; k++) {
                        const sliderDiv = document.createElement('div');
                        sliderDiv.className = 'weight-slider';
                        
                        const sliderId = `weight-${i}-${j}-${k}`;
                        sliderDiv.innerHTML = `
                            <label for="${sliderId}">W${i+1}[${j}][${k}]:</label>
                            <input type="range" id="${sliderId}" min="-2" max="2" step="0.01" value="${network.weights[i][j][k]}">
                            <span class="slider-value">${network.weights[i][j][k].toFixed(2)}</span>
                        `;
                        
                        const slider = sliderDiv.querySelector('input');
                        slider.addEventListener('input', function() {
                            network.weights[i][j][k] = parseFloat(this.value);
                            sliderDiv.querySelector('.slider-value').textContent = this.value;
                            renderNetwork();
                        });
                        
                        weightSlidersContainer.appendChild(sliderDiv);
                    }
                }
            }
            
            // Create bias sliders
            biasSlidersContainer.innerHTML = '';
            
            for (let i = 0; i < network.biases.length; i++) {
                const layerDiv = document.createElement('div');
                layerDiv.innerHTML = `<h4>Layer ${i + 1} Biases</h4>`;
                biasSlidersContainer.appendChild(layerDiv);
                
                for (let j = 0; j < network.biases[i].length; j++) {
                    const sliderDiv = document.createElement('div');
                    sliderDiv.className = 'bias-slider';
                    
                    const sliderId = `bias-${i}-${j}`;
                    sliderDiv.innerHTML = `
                        <label for="${sliderId}">B${i+1}[${j}]:</label>
                        <input type="range" id="${sliderId}" min="-2" max="2" step="0.01" value="${network.biases[i][j]}">
                        <span class="slider-value">${network.biases[i][j].toFixed(2)}</span>
                    `;
                    
                    const slider = sliderDiv.querySelector('input');
                    slider.addEventListener('input', function() {
                        network.biases[i][j] = parseFloat(this.value);
                        sliderDiv.querySelector('.slider-value').textContent = this.value;
                        renderNetwork();
                    });
                    
                    biasSlidersContainer.appendChild(sliderDiv);
                }
            }
        }
    }

    // Digit Recognizer Functions
    function setupDigitCanvas() {
        const ctx = digitCanvas.getContext('2d');
        let isDrawing = false;
        
        // Set canvas background to black
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, digitCanvas.width, digitCanvas.height);
        
        digitCanvas.addEventListener('mousedown', startDrawing);
        digitCanvas.addEventListener('mousemove', draw);
        digitCanvas.addEventListener('mouseup', stopDrawing);
        digitCanvas.addEventListener('mouseout', stopDrawing);
        
        // Touch support
        digitCanvas.addEventListener('touchstart', handleTouch);
        digitCanvas.addEventListener('touchmove', handleTouch);
        digitCanvas.addEventListener('touchend', stopDrawing);
        
        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent(
                e.type === 'touchstart' ? 'mousedown' : 'mousemove',
                {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                }
            );
            digitCanvas.dispatchEvent(mouseEvent);
        }
        
        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            const rect = digitCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(x, y, 10, 0, Math.PI * 2);
            ctx.fill();
        }
        
        function stopDrawing() {
            isDrawing = false;
        }
    }

    function clearDigitCanvas() {
        const ctx = digitCanvas.getContext('2d');
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, digitCanvas.width, digitCanvas.height);
        predictionOutput.textContent = '-';
        confidenceBar.innerHTML = '';
    }

    function predictDigit() {
        // Get the pixel data from canvas
        const ctx = digitCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, digitCanvas.width, digitCanvas.height);
        const data = imageData.data;
        
        // Convert to grayscale and downsample to 10x10
        const downsampled = new Array(10);
        for (let i = 0; i < 10; i++) {
            downsampled[i] = new Array(10);
            for (let j = 0; j < 10; j++) {
                let sum = 0;
                for (let y = i * 20; y < (i + 1) * 20; y++) {
                    for (let x = j * 20; x < (j + 1) * 20; x++) {
                        const idx = (y * digitCanvas.width + x) * 4;
                        sum += data[idx]; // R channel (grayscale)
                    }
                }
                downsampled[i][j] = sum / (20 * 20 * 255); // Normalize to 0-1
            }
        }
        
        // Flatten the 10x10 array to 100 elements
        const input = [];
        for (let i = 0; i < 10; i++) {
            for (let j = 0; j < 10; j++) {
                input.push(downsampled[i][j]);
            }
        }
        
        // For demo purposes, we'll use a simple "pretrained" model
        // In a real implementation, you would train a proper model
        const output = simpleDigitPredictor(input);
        
        // Display the prediction
        const predictedDigit = output.indexOf(Math.max(...output));
        predictionOutput.textContent = predictedDigit;
        
        // Show confidence bar
        confidenceBar.innerHTML = '';
        const confidenceLevel = document.createElement('div');
        confidenceLevel.className = 'confidence-level';
        confidenceLevel.style.width = `${output[predictedDigit] * 100}%`;
        confidenceBar.appendChild(confidenceLevel);
    }

    // Simple digit predictor (mock implementation)
    function simpleDigitPredictor(input) {
        // This is a mock function that returns random predictions
        // In a real implementation, you would use the neural network
        const output = new Array(10).fill(0);
        const randomDigit = Math.floor(Math.random() * 10);
        output[randomDigit] = Math.random() * 0.5 + 0.5; // Between 0.5 and 1.0
        
        // Make sure sum is 1 (simulating softmax)
        const sum = output.reduce((a, b) => a + b, 0);
        return output.map(val => val / sum);
    }

    // Gradient Descent Visualization
    function setupGradientDescent() {
        const ctx = gradientCanvas.getContext('2d');
        const width = gradientCanvas.width;
        const height = gradientCanvas.height;
        
        // Draw the initial function
        drawFunction();
        
        function drawFunction() {
            ctx.clearRect(0, 0, width, height);
            
            // Draw axes
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, height / 2);
            ctx.lineTo(width, height / 2);
            ctx.moveTo(width / 2, 0);
            ctx.lineTo(width / 2, height);
            ctx.stroke();
            
            // Draw function (x^2)
            ctx.strokeStyle = '#6e48aa';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let x = -width / 2; x < width / 2; x += 0.1) {
                const xPos = x + width / 2;
                const yPos = height / 2 - (x * x) / 10;
                
                if (x === -width / 2) {
                    ctx.moveTo(xPos, yPos);
                } else {
                    ctx.lineTo(xPos, yPos);
                }
            }
            
            ctx.stroke();
        }
    }

    function startGradientDescent() {
        const ctx = gradientCanvas.getContext('2d');
        const width = gradientCanvas.width;
        const height = gradientCanvas.height;
        
        // Initial position (random x)
        let x = Math.random() * 4 - 2; // Between -2 and 2
        let y = x * x;
        
        // Learning rate for visualization
        const lr = 0.1;
        
        // Draw the initial point
        drawFunction();
        drawPoint(x, y);
        
        // Animate gradient descent
        let animationId;
        let iteration = 0;
        const maxIterations = 20;
        
        function animate() {
            if (iteration >= maxIterations) {
                cancelAnimationFrame(animationId);
                return;
            }
            
            // Calculate gradient (derivative of x^2 is 2x)
            const gradient = 2 * x;
            
            // Update position
            x -= lr * gradient;
            y = x * x;
            
            // Redraw
            drawFunction();
            drawPoint(x, y);
            
            iteration++;
            animationId = requestAnimationFrame(animate);
        }
        
        animate();
        
        function drawPoint(xVal, yVal) {
            const xPos = xVal * (width / 4) + width / 2;
            const yPos = height / 2 - yVal * (height / 40);
            
            ctx.fillStyle = '#ff7e5f';
            ctx.beginPath();
            ctx.arc(xPos, yPos, 5, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw tangent line
            const gradient = 2 * xVal;
            const tangentLength = 50;
            const x1 = xPos - tangentLength / Math.sqrt(1 + gradient * gradient);
            const y1 = yPos - gradient * tangentLength / Math.sqrt(1 + gradient * gradient);
            const x2 = xPos + tangentLength / Math.sqrt(1 + gradient * gradient);
            const y2 = yPos + gradient * tangentLength / Math.sqrt(1 + gradient * gradient);
            
            ctx.strokeStyle = '#f44336';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
    }

    function resetGradientDescent() {
        setupGradientDescent();
    }
});