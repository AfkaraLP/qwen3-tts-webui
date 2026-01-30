/**
 * Main application logic for Qwen TTS WebUI
 * Coordinates between all modules
 */

// Global state
window.selectedReferenceId = null;
window.currentTaskId = null;
window.progressInterval = null;
window.elapsedTimeInterval = null;
window.generationStartTime = null;
window.currentAudioSource = 'file';
window.referencesData = {};
window.availableReferences = [];

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', function() {
    loadReferences();
    loadLanguages();
    loadGeneratedAudios();
});

/**
 * Load available reference audios
 */
async function loadReferences() {
    try {
        const references = await fetchReferences();
        
        const selectElement = document.getElementById('reference-select');
        
        // Clear existing options except the default one
        selectElement.innerHTML = '<option value="">-- Choose a reference audio --</option>';
        
        if (references.length === 0) {
            selectElement.innerHTML = '<option value="">No reference audios found. Upload one first!</option>';
            return;
        }
        
        references.forEach(ref => {
            const option = document.createElement('option');
            option.value = ref.id;
            option.textContent = ref.name || ref.original_name;
            selectElement.appendChild(option);
        });
        
        // Store references for preview functionality
        window.availableReferences = references;
        
        // Store references data for rename functionality
        window.referencesData = {};
        references.forEach(ref => {
            window.referencesData[ref.id] = ref;
        });
        
        // Update multi-speaker dropdowns
        refreshAllSegmentDropdowns();
        
    } catch (error) {
        console.error('Error loading references:', error);
        showError('Failed to load reference audios');
    }
}

/**
 * Load available languages
 */
async function loadLanguages() {
    try {
        const data = await fetchLanguages();
        const languages = data.languages;
        
        // Populate existing reference language dropdown
        const existingSelect = document.getElementById('language-existing');
        existingSelect.innerHTML = '';
        languages.forEach(lang => {
            const option = document.createElement('option');
            option.value = lang;
            option.textContent = lang.charAt(0).toUpperCase() + lang.slice(1);
            if (lang === 'English') {
                option.selected = true;
            }
            existingSelect.appendChild(option);
        });
        
        // Populate upload language dropdown
        const uploadSelect = document.getElementById('language-upload');
        uploadSelect.innerHTML = '';
        languages.forEach(lang => {
            const option = document.createElement('option');
            option.value = lang;
            option.textContent = lang.charAt(0).toUpperCase() + lang.slice(1);
            if (lang === 'English') {
                option.selected = true;
            }
            uploadSelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading languages:', error);
        showError('Failed to load languages');
    }
}

/**
 * Load generated audios list
 */
async function loadGeneratedAudios() {
    try {
        const generatedAudios = await fetchGeneratedAudios();
        
        const container = document.getElementById('generated-audios-list');
        
        if (generatedAudios.length === 0) {
            container.innerHTML = '<p>No generated audios found. Create your first voice clone!</p>';
            return;
        }
        
        container.innerHTML = '';
        generatedAudios.forEach(audio => {
            const audioDiv = document.createElement('div');
            audioDiv.className = 'generated-audio-item';
            
            const titleDiv = document.createElement('div');
            titleDiv.className = 'generated-audio-title';
            titleDiv.textContent = audio.generated_text.substring(0, 80) + (audio.generated_text.length > 80 ? '...' : '');
            
            const infoDiv = document.createElement('div');
            infoDiv.className = 'generated-audio-meta';
            const generationTimeText = audio.generation_time_seconds 
                ? `<br>Generation time: ${audio.generation_time_seconds}s` 
                : '';
            infoDiv.innerHTML = `
                Reference: ${audio.ref_audio_name}<br>
                Created: ${new Date(parseFloat(audio.created_at) * 1000).toLocaleString()}${generationTimeText}
            `;
            
            const controlsDiv = document.createElement('div');
            controlsDiv.className = 'generated-audio-controls';
            
            const audioControl = document.createElement('audio');
            audioControl.controls = true;
            audioControl.src = `/tasks/${audio.id}/audio`;
            
            const downloadBtn = document.createElement('a');
            downloadBtn.href = `/tasks/${audio.id}/audio`;
            downloadBtn.download = `cloned_voice_${audio.id}.wav`;
            downloadBtn.className = 'btn';
            downloadBtn.textContent = 'Download';
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn btn-danger';
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = () => deleteGeneratedAudio(audio.id);
            
            controlsDiv.appendChild(audioControl);
            controlsDiv.appendChild(downloadBtn);
            controlsDiv.appendChild(deleteBtn);
            
            audioDiv.appendChild(titleDiv);
            audioDiv.appendChild(infoDiv);
            audioDiv.appendChild(controlsDiv);
            container.appendChild(audioDiv);
        });
        
    } catch (error) {
        console.error('Error loading generated audios:', error);
        showError('Failed to load generated audios');
    }
}

/**
 * Delete a generated audio
 * @param {string} audioId - Audio ID to delete
 */
async function deleteGeneratedAudio(audioId) {
    try {
        const response = await deleteGeneratedAudioApi(audioId);
        
        if (response.ok) {
            loadGeneratedAudios();
        } else {
            const error = await response.json();
            showError(error.detail || 'Failed to delete audio');
        }
    } catch (error) {
        console.error('Error deleting generated audio:', error);
        showError('Failed to delete audio');
    }
}

/**
 * Handle reference selection from dropdown
 */
function selectReferenceFromDropdown() {
    const selectElement = document.getElementById('reference-select');
    window.selectedReferenceId = selectElement.value;
    
    const previewDiv = document.getElementById('reference-preview');
    const previewAudio = document.getElementById('preview-audio');
    const renameDiv = document.getElementById('reference-rename');
    const nameInput = document.getElementById('reference-name-input');
    
    if (window.selectedReferenceId) {
        // Show preview and load audio
        previewDiv.style.display = 'block';
        previewAudio.src = `/references/${window.selectedReferenceId}/audio`;
        
        // Show rename section and populate with current name
        renameDiv.style.display = 'block';
        const refData = window.referencesData[window.selectedReferenceId];
        nameInput.value = refData.name || refData.original_name;
    } else {
        // Hide preview and rename sections
        previewDiv.style.display = 'none';
        renameDiv.style.display = 'none';
        previewAudio.src = '';
    }
}

/**
 * Rename a reference audio
 */
async function renameReference() {
    if (!window.selectedReferenceId) {
        showError('Please select a reference audio first');
        return;
    }
    
    const nameInput = document.getElementById('reference-name-input');
    const newName = nameInput.value.trim();
    
    if (!newName) {
        showError('Please enter a name for the reference');
        return;
    }
    
    try {
        await renameReferenceApi(window.selectedReferenceId, newName);
        
        // Update the stored data
        if (window.referencesData[window.selectedReferenceId]) {
            window.referencesData[window.selectedReferenceId].name = newName;
        }
        
        // Update the dropdown display
        const selectElement = document.getElementById('reference-select');
        const option = selectElement.querySelector(`option[value="${window.selectedReferenceId}"]`);
        if (option) {
            option.textContent = newName;
        }
        
        showSuccess('Reference renamed successfully');
        
    } catch (error) {
        console.error('Error renaming reference:', error);
        showError('Failed to rename reference');
    }
}

/**
 * Clone with existing reference
 */
async function cloneWithExistingReference() {
    if (!window.selectedReferenceId) {
        showError('Please select a reference audio from the dropdown');
        return;
    }
    
    const text = document.getElementById('text-existing').value;
    const language = document.getElementById('language-existing').value;
    
    if (!text.trim()) {
        showError('Please enter text to generate');
        return;
    }
    
    await performCloning('/clone', {
        text: text,
        ref_audio_id: window.selectedReferenceId,
        language: language
    });
}

/**
 * Clone with uploaded audio
 */
async function cloneWithUpload() {
    const text = document.getElementById('text-upload').value;
    const language = document.getElementById('language-upload').value;
    
    if (!text.trim()) {
        showError('Please enter text to generate');
        return;
    }
    
    if (window.currentAudioSource === 'youtube') {
        const youtubeUrl = document.getElementById('youtube-url').value;
        const youtubeName = document.getElementById('youtube-name').value;
        
        if (!youtubeUrl.trim()) {
            showError('Please enter a YouTube URL');
            return;
        }
        
        await performCloning('/clone-with-youtube', {
            youtube_url: youtubeUrl,
            text: text,
            language: language,
            name: youtubeName || undefined
        });
    } else {
        let file;
        
        if (window.currentAudioSource === 'file') {
            const fileInput = document.getElementById('reference-file');
            file = fileInput.files[0];
            
            if (!file) {
                showError('Please select a reference audio file');
                return;
            }
        } else { // record
            const recordedBlob = getRecordedBlob();
            if (!recordedBlob) {
                showError('Please record audio first');
                return;
            }
            file = createFileFromRecording();
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('text', text);
        formData.append('language', language);
        
        await performCloning('/clone-with-upload', formData, true);
    }
}

/**
 * Clone multi-speaker audio
 */
async function cloneMultiSpeaker() {
    const validation = validateMultiSpeakerSegments();
    
    if (!validation.valid) {
        showError(validation.error);
        return;
    }
    
    await performCloning('/clone-multi-speaker', { segments: validation.segments });
}

/**
 * Perform the cloning operation
 * @param {string} endpoint - API endpoint
 * @param {Object|FormData} data - Request data
 * @param {boolean} isFormData - Whether data is FormData
 */
async function performCloning(endpoint, data, isFormData = false) {
    hideAllSections();
    
    const progressSection = document.getElementById('progress-section');
    progressSection.classList.remove('hidden');
    
    // Start elapsed time counter
    startElapsedTimeCounter();
    
    try {
        let response;
        if (isFormData) {
            response = await fetch(endpoint, {
                method: 'POST',
                body: data
            });
        } else {
            response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
        }
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to start cloning');
        }
        
        const result = await response.json();
        window.currentTaskId = result.task_id;
        
        // Start polling for progress
        startProgressPolling();
        
    } catch (error) {
        console.error('Error starting cloning:', error);
        showError(error.message);
        stopElapsedTimeCounter();
    }
}

/**
 * Start polling for task progress
 */
function startProgressPolling() {
    if (window.progressInterval) {
        clearInterval(window.progressInterval);
    }
    
    window.progressInterval = setInterval(async () => {
        try {
            const task = await fetchTaskStatus(window.currentTaskId);
            
            // Use appropriate progress update based on task type
            if (task.is_multi_speaker) {
                updateProgressMultiSpeaker(task);
            } else {
                updateProgress(task);
            }
            
            if (task.status === 'completed') {
                clearInterval(window.progressInterval);
                stopElapsedTimeCounter();
                showResult(task.output_path, task.generation_time_seconds);
                loadGeneratedAudios(); // Refresh the generated audios list
            } else if (task.status === 'failed') {
                clearInterval(window.progressInterval);
                stopElapsedTimeCounter();
                hideAllSections();
                showError(task.error || 'Cloning failed');
            } else if (task.status === 'cancelled') {
                clearInterval(window.progressInterval);
                stopElapsedTimeCounter();
                hideAllSections();
                document.getElementById('error-section').classList.remove('hidden');
                document.getElementById('error-message').textContent = 'Generation was cancelled';
            }
        } catch (error) {
            console.error('Error polling task status:', error);
        }
    }, 1000);
}

/**
 * Cancel the current generation
 */
async function cancelGeneration() {
    if (!window.currentTaskId) {
        return;
    }
    
    try {
        const response = await cancelTaskApi(window.currentTaskId);
        
        // Always stop the animation and hide progress section when cancel is clicked
        clearInterval(window.progressInterval);
        stopElapsedTimeCounter();
        hideAllSections();
        
        if (response.ok) {
            window.currentTaskId = null;
            showSuccess('Generation cancelled');
        } else {
            // Task may have already completed or failed, still hide the progress
            const error = await response.json();
            window.currentTaskId = null;
            // Don't show error if task already finished - just a minor notification
            console.log('Cancel response:', error.detail);
        }
    } catch (error) {
        // Even on network error, stop the animation
        clearInterval(window.progressInterval);
        stopElapsedTimeCounter();
        hideAllSections();
        console.error('Error cancelling task:', error);
        showError('Failed to cancel generation');
    }
}
