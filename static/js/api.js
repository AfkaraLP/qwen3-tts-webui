/**
 * API functions for Qwen TTS WebUI
 */

/**
 * Fetch all available reference audios
 * @returns {Promise<Array>} Array of reference audio objects
 */
async function fetchReferences() {
    const response = await fetch('/references');
    return await response.json();
}

/**
 * Fetch available languages
 * @returns {Promise<Object>} Object containing languages array
 */
async function fetchLanguages() {
    const response = await fetch('/languages');
    return await response.json();
}

/**
 * Fetch all generated audios
 * @returns {Promise<Array>} Array of generated audio objects
 */
async function fetchGeneratedAudios() {
    const response = await fetch('/generated');
    return await response.json();
}

/**
 * Delete a generated audio
 * @param {string} audioId - ID of the audio to delete
 * @returns {Promise<Response>} Fetch response
 */
async function deleteGeneratedAudioApi(audioId) {
    return await fetch(`/generated/${audioId}`, {
        method: 'DELETE'
    });
}

/**
 * Rename a reference audio
 * @param {string} referenceId - ID of the reference
 * @param {string} newName - New name for the reference
 * @returns {Promise<Object>} API response
 */
async function renameReferenceApi(referenceId, newName) {
    const response = await fetch(`/references/${referenceId}/name`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: newName })
    });
    
    if (!response.ok) {
        throw new Error('Failed to rename reference');
    }
    
    return await response.json();
}

/**
 * Get task status
 * @param {string} taskId - Task ID
 * @returns {Promise<Object>} Task status object
 */
async function fetchTaskStatus(taskId) {
    const response = await fetch(`/tasks/${taskId}`);
    return await response.json();
}

/**
 * Cancel a task
 * @param {string} taskId - Task ID
 * @returns {Promise<Response>} Fetch response
 */
async function cancelTaskApi(taskId) {
    return await fetch(`/tasks/${taskId}/cancel`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });
}

/**
 * Clone voice with existing reference
 * @param {Object} data - Clone data {text, ref_audio_id, language}
 * @returns {Promise<Object>} API response with task_id
 */
async function cloneWithExistingReferenceApi(data) {
    const response = await fetch('/clone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start cloning');
    }
    
    return await response.json();
}

/**
 * Clone voice with YouTube URL
 * @param {Object} data - Clone data {youtube_url, text, language, name}
 * @returns {Promise<Object>} API response with task_id
 */
async function cloneWithYoutubeApi(data) {
    const response = await fetch('/clone-with-youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start cloning');
    }
    
    return await response.json();
}

/**
 * Clone voice with uploaded file
 * @param {FormData} formData - Form data with file, text, language
 * @returns {Promise<Object>} API response with task_id
 */
async function cloneWithUploadApi(formData) {
    const response = await fetch('/clone-with-upload', {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start cloning');
    }
    
    return await response.json();
}

/**
 * Clone multi-speaker audio
 * @param {Object} data - Clone data {segments: [{text, ref_audio_id, language}, ...]}
 * @returns {Promise<Object>} API response with task_id
 */
async function cloneMultiSpeakerApi(data) {
    const response = await fetch('/clone-multi-speaker', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start cloning');
    }
    
    return await response.json();
}
