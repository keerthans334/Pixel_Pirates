// Get modal elements
const trainModal = document.getElementById("trainModal");
const recognizeModal = document.getElementById("recognizeModal");
const listenModal = document.getElementById("listenModal");

// Get buttons
const trainBtn = document.getElementById("trainBtn");
const recognizeBtn = document.getElementById("recognizeBtn");
const listenBtn = document.getElementById("listenBtn");
const clearCacheBtn = document.getElementById("clearCacheBtn");

// Get close buttons
const closeTrainModal = document.getElementById("closeTrainModal");
const closeRecognizeModal = document.getElementById("closeRecognizeModal");
const closeListenModal = document.getElementById("closeListenModal");

// Open modals on button click
trainBtn.addEventListener("click", () => {
    trainModal.style.display = "block";
});

recognizeBtn.addEventListener("click", () => {
    recognizeModal.style.display = "block";
});

listenBtn.addEventListener("click", () => {
    listenModal.style.display = "block";
});

// Close modals when the close button is clicked
closeTrainModal.addEventListener("click", () => {
    trainModal.style.display = "none";
});

closeRecognizeModal.addEventListener("click", () => {
    recognizeModal.style.display = "none";
});

closeListenModal.addEventListener("click", () => {
    listenModal.style.display = "none";
});

// Function to clear cache
clearCacheBtn.addEventListener("click", () => {
    if (confirm("Are you sure you want to clear the cache?")) {
        // Simulate cache clearing process
        alert("Cache cleared successfully!");
    }
});

// You can extend this by adding the functionalities of each button based on your backend and API calls

