// Penguins of Madagascar Classifier - JavaScript

// Function to load the latest prediction
async function loadLatestPrediction() {
    try {
        const response = await fetch('latest_prediction.json');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        displayLatestPrediction(data);
    } catch (error) {
        console.error('Error loading latest prediction:', error);
    }
}

// Function to load prediction history
async function loadPredictionHistory() {
    try {
        const response = await fetch('prediction_history.json');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        displayPredictionHistory(data);
    } catch (error) {
        console.error('Error loading prediction history:', error);
    }
}

// Function to display the latest prediction
function displayLatestPrediction(prediction) {
    // Update species and timestamp
    document.querySelector('.card-body h4 .badge').textContent = prediction.predicted_species;
    document.querySelector('.card-body h4 .badge').className = `badge bg-${getSpeciesColor(prediction.predicted_species)}`;
    document.querySelector('.card-body > p').textContent = `Timestamp: ${prediction.timestamp}`;
    
    // Update measurement data
    const listItems = document.querySelectorAll('.list-group-item');
    listItems[0].textContent = `Bill Length: ${prediction.penguin_data.bill_length_mm} mm`;
    listItems[1].textContent = `Bill Depth: ${prediction.penguin_data.bill_depth_mm} mm`;
    listItems[2].textContent = `Flipper Length: ${prediction.penguin_data.flipper_length_mm} mm`;
    listItems[3].textContent = `Body Mass: ${prediction.penguin_data.body_mass_g} g`;
    
    // Update prediction confidence bars
    updateProgressBar('Adelie', prediction.probabilities.Adelie);
    updateProgressBar('Chinstrap', prediction.probabilities.Chinstrap);
    updateProgressBar('Gentoo', prediction.probabilities.Gentoo);
    
    // Update last updated text
    document.querySelector('.card-body p:last-child').textContent = `Last updated: ${prediction.timestamp}`;
}

// Function to display prediction history
function displayPredictionHistory(history) {
    const historyContainer = document.querySelector('.history-container');
    historyContainer.innerHTML = '';
    
    history.forEach(pred => {
        const historyItem = document.createElement('div');
        historyItem.className = `history-item ${pred.predicted_species.toLowerCase()}`;
        
        historyItem.innerHTML = `
            <strong>Species:</strong> ${pred.predicted_species} 
            <strong>Time:</strong> ${pred.timestamp}
            <br>
            <small>Bill: ${pred.penguin_data.bill_length_mm}mm × ${pred.penguin_data.bill_depth_mm}mm, 
            Flipper: ${pred.penguin_data.flipper_length_mm}mm, 
            Mass: ${pred.penguin_data.body_mass_g}g</small>
        `;
        
        historyContainer.appendChild(historyItem);
    });
}

// Helper function to update progress bars
function updateProgressBar(species, probability) {
    const percentage = probability * 100;
    const progressBar = document.querySelector(`.progress-bar:nth-of-type(${getSpeciesIndex(species)})`);
    
    progressBar.style.width = `${percentage}%`;
    progressBar.setAttribute('aria-valuenow', percentage);
    progressBar.textContent = `${species}: ${percentage.toFixed(1)}%`;
}

// Helper function to get species color class
function getSpeciesColor(species) {
    switch(species) {
        case 'Adelie': return 'success';
        case 'Chinstrap': return 'info';
        case 'Gentoo': return 'warning';
        default: return 'primary';
    }
}

// Helper function to get species index for progress bars
function getSpeciesIndex(species) {
    switch(species) {
        case 'Adelie': return 1;
        case 'Chinstrap': return 2;
        case 'Gentoo': return 3;
        default: return 1;
    }
}

// Load data when the page loads
document.addEventListener('DOMContentLoaded', () => {
    loadLatestPrediction();
    loadPredictionHistory();
});
