/**
 * Audio recording functionality for Qwen TTS WebUI
 */

// Recording state variables
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;

/**
 * Get the recorded audio blob
 * @returns {Blob|null} The recorded audio blob or null
 */
function getRecordedBlob() {
    return recordedBlob;
}

/**
 * Clear the recorded blob
 */
function clearRecordedBlob() {
    recordedBlob = null;
}

/**
 * Toggle recording on/off
 */
async function toggleRecording() {
    try {
        // Check if mediaDevices API is available (requires secure context)
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
            if (!isLocalhost && window.location.protocol !== 'https:') {
                showError('Microphone access requires HTTPS. Please access this page via localhost or HTTPS.');
            } else {
                showError('Your browser does not support audio recording.');
            }
            return;
        }

        // Use more flexible audio constraints for better browser compatibility
        const audioConstraints = {
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        };

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia(audioConstraints);
        } catch (constraintError) {
            // If constraints fail, try with simple audio: true
            console.warn('Detailed constraints failed, trying simple constraints:', constraintError);
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        
        // Determine the best supported MIME type for recording
        // Firefox prefers ogg, Chrome/Safari prefer webm
        let mimeType = '';
        const mimeTypes = [
            'audio/ogg;codecs=opus',  // Firefox preferred
            'audio/webm;codecs=opus', // Chrome preferred
            'audio/ogg',
            'audio/webm',
            'audio/mp4',
            'audio/mpeg'
        ];
        
        for (const type of mimeTypes) {
            if (MediaRecorder.isTypeSupported(type)) {
                mimeType = type;
                break;
            }
        }
        
        // Create MediaRecorder with or without mimeType option
        const options = mimeType ? { mimeType } : {};
        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            // Use the actual MIME type from the MediaRecorder
            const actualMimeType = mediaRecorder.mimeType || mimeType || 'audio/ogg';
            recordedBlob = new Blob(audioChunks, { type: actualMimeType });
            const audioUrl = URL.createObjectURL(recordedBlob);
            const recordedAudio = document.getElementById('recorded-audio');
            recordedAudio.src = audioUrl;
            recordedAudio.style.display = 'block';
        };

        mediaRecorder.start();
        
        document.getElementById('record-btn').style.display = 'none';
        document.getElementById('stop-btn').style.display = 'inline-block';
        document.getElementById('recording-status').textContent = 'Recording... Click Stop when finished.';
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        console.error('Error name:', error.name);
        console.error('Error message:', error.message);
        
        // Provide more specific error messages
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            showError('Microphone permission denied. Please allow microphone access in your browser settings and reload the page.');
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            // Additional help for Firefox on macOS
            showError('No microphone found. On macOS, please check: System Preferences > Security & Privacy > Privacy > Microphone, and ensure Firefox is allowed.');
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            showError('Microphone is in use by another application. Please close other apps using the microphone.');
        } else if (error.name === 'OverconstrainedError') {
            showError('Could not satisfy audio constraints. Please try a different microphone.');
        } else if (error.name === 'SecurityError') {
            showError('Microphone access blocked due to security policy. Please use HTTPS or localhost.');
        } else if (error.name === 'AbortError') {
            showError('Microphone access was aborted. Please try again.');
        } else {
            showError(`Could not access microphone: ${error.message || error.name || 'Unknown error'}. Check browser console for details.`);
        }
    }
}

/**
 * Stop the current recording
 */
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        document.getElementById('record-btn').style.display = 'inline-block';
        document.getElementById('stop-btn').style.display = 'none';
        document.getElementById('recording-status').textContent = 'Recording complete! Audio ready for upload.';
    }
}

/**
 * Get file extension mapping for MIME types
 * @returns {Object} MIME type to extension mapping
 */
function getMimeToExtMap() {
    return {
        'audio/webm': 'webm',
        'audio/webm;codecs=opus': 'webm',
        'audio/ogg': 'ogg',
        'audio/ogg;codecs=opus': 'ogg',
        'audio/mp4': 'mp4',
        'audio/mpeg': 'mp3',
        'audio/wav': 'wav'
    };
}

/**
 * Create a File object from the recorded blob
 * @returns {File|null} File object or null if no recording
 */
function createFileFromRecording() {
    if (!recordedBlob) {
        return null;
    }
    
    const mimeToExt = getMimeToExtMap();
    const ext = mimeToExt[recordedBlob.type] || 'webm';
    return new File([recordedBlob], `recorded_audio.${ext}`, { type: recordedBlob.type });
}
