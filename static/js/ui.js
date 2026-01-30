/**
 * UI/DOM manipulation functions for Qwen TTS WebUI
 */

/**
 * Switch between main tabs (existing, upload, multispeaker)
 * @param {string} tab - Tab identifier
 */
function switchTab(tab) {
    // Update tab buttons
    document.querySelectorAll('.tabs .tab').forEach(t => t.classList.remove('active'));
    const tabIndex = tab === 'existing' ? 1 : tab === 'upload' ? 2 : 3;
    document.querySelector(`.tabs .tab:nth-child(${tabIndex})`).classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.card > .tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`${tab}-tab`).classList.add('active');
    
    // Initialize multi-speaker tab if switching to it for the first time
    if (tab === 'multispeaker') {
        initializeMultiSpeakerTab();
    }
    
    // Hide other sections
    hideAllSections();
}

/**
 * Switch audio source in upload tab (file, record, youtube)
 * @param {string} source - Source identifier
 */
function switchAudioSource(source) {
    window.currentAudioSource = source;
    
    // Update tab buttons in upload section
    const uploadTabs = document.querySelectorAll('.audio-upload-tabs .tab');
    uploadTabs.forEach(t => t.classList.remove('active'));
    const tabIndex = source === 'file' ? 0 : source === 'record' ? 1 : 2;
    uploadTabs[tabIndex].classList.add('active');
    
    // Update tab content
    document.querySelectorAll('#upload-tab .tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`${source}-source`).classList.add('active');
}

/**
 * Hide all result/progress/error sections
 */
function hideAllSections() {
    document.getElementById('progress-section').classList.add('hidden');
    document.getElementById('result-section').classList.add('hidden');
    document.getElementById('error-section').classList.add('hidden');
}

/**
 * Show a toast notification
 * @param {string} message - Message to display
 * @param {string} type - Toast type ('error' or 'success')
 * @param {number} duration - Duration in milliseconds
 */
function showToast(message, type = 'error', duration = 5000) {
    const container = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    // Slide out and remove after duration
    setTimeout(() => {
        toast.classList.add('slide-out');
        // Remove from DOM after animation completes
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, duration);
}

/**
 * Show error toast
 * @param {string} message - Error message
 */
function showError(message) {
    showToast(message, 'error', 5000);
}

/**
 * Show success toast
 * @param {string} message - Success message
 */
function showSuccess(message) {
    showToast(message, 'success', 3000);
}

/**
 * Update progress bar and text
 * @param {Object} task - Task object with progress and status
 */
function updateProgress(task) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    const progress = task.progress || 0;
    progressFill.style.width = `${progress}%`;
    
    if (task.status === 'processing') {
        if (progress < 25) {
            progressText.textContent = 'Initializing voice cloner...';
        } else if (progress < 50) {
            progressText.textContent = 'Processing reference audio...';
        } else if (progress < 75) {
            progressText.textContent = 'Generating cloned voice...';
        } else {
            progressText.textContent = 'Finalizing audio...';
        }
    } else if (task.status === 'cancelled') {
        progressText.textContent = 'Generation cancelled';
        if (window.progressInterval) {
            clearInterval(window.progressInterval);
        }
        stopElapsedTimeCounter();
        hideAllSections();
        document.getElementById('error-section').classList.remove('hidden');
        document.getElementById('error-message').textContent = 'Generation was cancelled';
    }
}

/**
 * Update progress for multi-speaker tasks
 * @param {Object} task - Task object
 */
function updateProgressMultiSpeaker(task) {
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    const progress = task.progress || 0;
    progressFill.style.width = `${progress}%`;
    
    if (task.is_multi_speaker && task.total_segments) {
        const currentSeg = task.current_segment || 0;
        const totalSeg = task.total_segments;
        
        if (progress < 5) {
            progressText.textContent = 'Initializing multi-speaker generation...';
        } else if (progress < 90) {
            progressText.textContent = `Generating segment ${currentSeg} of ${totalSeg}...`;
        } else if (progress < 95) {
            progressText.textContent = 'Concatenating audio segments...';
        } else {
            progressText.textContent = 'Finalizing audio...';
        }
    }
}

/**
 * Show result section with generated audio
 * @param {string} outputPath - Path to output file
 * @param {number} generationTimeSeconds - Generation time in seconds
 */
function showResult(outputPath, generationTimeSeconds) {
    hideAllSections();
    
    const resultSection = document.getElementById('result-section');
    const resultAudio = document.getElementById('result-audio');
    const downloadLink = document.getElementById('download-link');
    const generationTimeDisplay = document.getElementById('result-generation-time');
    
    resultAudio.src = `/tasks/${window.currentTaskId}/audio`;
    downloadLink.href = `/tasks/${window.currentTaskId}/audio`;
    
    // Display generation time
    if (generationTimeSeconds) {
        generationTimeDisplay.textContent = `Generated in ${generationTimeSeconds} seconds`;
        generationTimeDisplay.style.display = 'block';
    } else {
        generationTimeDisplay.style.display = 'none';
    }
    
    resultSection.classList.remove('hidden');
}

/**
 * Populate a select element with options
 * @param {HTMLSelectElement} selectElement - Select element to populate
 * @param {Array} options - Array of {value, text} objects
 * @param {string} defaultText - Default option text
 * @param {string} selectedValue - Value to select by default
 */
function populateSelect(selectElement, options, defaultText = '', selectedValue = null) {
    selectElement.innerHTML = defaultText ? `<option value="">${defaultText}</option>` : '';
    
    options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.text;
        if (selectedValue && opt.value === selectedValue) {
            option.selected = true;
        }
        selectElement.appendChild(option);
    });
}

/**
 * Start elapsed time counter
 */
function startElapsedTimeCounter() {
    window.generationStartTime = Date.now();
    const elapsedTimeDisplay = document.getElementById('elapsed-time');
    elapsedTimeDisplay.textContent = 'Elapsed: 0s';
    
    if (window.elapsedTimeInterval) {
        clearInterval(window.elapsedTimeInterval);
    }
    
    window.elapsedTimeInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - window.generationStartTime) / 1000);
        elapsedTimeDisplay.textContent = `Elapsed: ${elapsed}s`;
    }, 1000);
}

/**
 * Stop elapsed time counter
 */
function stopElapsedTimeCounter() {
    if (window.elapsedTimeInterval) {
        clearInterval(window.elapsedTimeInterval);
        window.elapsedTimeInterval = null;
    }
}
