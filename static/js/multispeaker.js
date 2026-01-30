/**
 * Multi-speaker functionality for Qwen TTS WebUI
 */

// Multi-speaker state
let speakerSegmentCount = 0;
let multiSpeakerInitialized = false;

/**
 * Initialize the multi-speaker tab with default segments
 */
function initializeMultiSpeakerTab() {
    if (multiSpeakerInitialized) return;
    
    // Add two initial segments
    addSpeakerSegment();
    addSpeakerSegment();
    
    multiSpeakerInitialized = true;
}

/**
 * Add a new speaker segment
 */
function addSpeakerSegment() {
    speakerSegmentCount++;
    const segmentId = speakerSegmentCount;
    const container = document.getElementById('speaker-segments-container');
    
    const segmentDiv = document.createElement('div');
    segmentDiv.className = 'speaker-segment';
    segmentDiv.id = `segment-${segmentId}`;
    segmentDiv.innerHTML = `
        <div class="speaker-segment-header">
            <span class="speaker-segment-title">Segment ${segmentId}</span>
            <button class="speaker-segment-remove" onclick="removeSpeakerSegment(${segmentId})">Remove</button>
        </div>
        <div class="segment-row">
            <div class="form-group">
                <label for="segment-${segmentId}-speaker">Reference Speaker:</label>
                <select id="segment-${segmentId}-speaker" class="segment-speaker-select">
                    <option value="">-- Choose a speaker --</option>
                </select>
            </div>
            <div class="form-group">
                <label for="segment-${segmentId}-language">Language:</label>
                <select id="segment-${segmentId}-language" class="segment-language-select">
                    <option value="auto">Auto</option>
                </select>
            </div>
        </div>
        <div class="form-group">
            <label for="segment-${segmentId}-text">Text to Generate:</label>
            <textarea id="segment-${segmentId}-text" class="segment-text" placeholder="Enter the text for this speaker..." rows="2"></textarea>
        </div>
    `;
    
    container.appendChild(segmentDiv);
    
    // Populate the speaker dropdown
    populateSegmentSpeakerDropdown(segmentId);
    
    // Populate the language dropdown
    populateSegmentLanguageDropdown(segmentId);
    
    updateSegmentNumbers();
}

/**
 * Remove a speaker segment
 * @param {number} segmentId - ID of segment to remove
 */
function removeSpeakerSegment(segmentId) {
    const segment = document.getElementById(`segment-${segmentId}`);
    if (segment) {
        segment.remove();
        updateSegmentNumbers();
    }
}

/**
 * Update segment numbers after add/remove
 */
function updateSegmentNumbers() {
    const segments = document.querySelectorAll('.speaker-segment');
    segments.forEach((segment, index) => {
        const title = segment.querySelector('.speaker-segment-title');
        if (title) {
            title.textContent = `Segment ${index + 1}`;
        }
    });
}

/**
 * Populate speaker dropdown for a segment
 * @param {number} segmentId - Segment ID
 */
function populateSegmentSpeakerDropdown(segmentId) {
    const select = document.getElementById(`segment-${segmentId}-speaker`);
    if (!select || !window.availableReferences) return;
    
    // Preserve the currently selected value
    const currentValue = select.value;
    
    select.innerHTML = '<option value="">-- Choose a speaker --</option>';
    window.availableReferences.forEach(ref => {
        const option = document.createElement('option');
        option.value = ref.id;
        option.textContent = ref.name || ref.original_name;
        select.appendChild(option);
    });
    
    // Restore the previously selected value if it still exists
    if (currentValue) {
        select.value = currentValue;
    }
}

/**
 * Populate language dropdown for a segment
 * @param {number} segmentId - Segment ID
 */
function populateSegmentLanguageDropdown(segmentId) {
    const select = document.getElementById(`segment-${segmentId}-language`);
    if (!select) return;
    
    // Try to get languages from the existing dropdown
    const existingLanguageSelect = document.getElementById('language-existing');
    if (existingLanguageSelect && existingLanguageSelect.options.length > 0) {
        select.innerHTML = '';
        for (const option of existingLanguageSelect.options) {
            const newOption = document.createElement('option');
            newOption.value = option.value;
            newOption.textContent = option.textContent;
            if (option.value === 'English') {
                newOption.selected = true;
            }
            select.appendChild(newOption);
        }
    }
}

/**
 * Get all multi-speaker segments data
 * @returns {Array} Array of segment objects
 */
function getMultiSpeakerSegments() {
    const segments = [];
    const segmentElements = document.querySelectorAll('.speaker-segment');
    
    segmentElements.forEach((element, index) => {
        const speakerSelect = element.querySelector('.segment-speaker-select');
        const languageSelect = element.querySelector('.segment-language-select');
        const textArea = element.querySelector('.segment-text');
        
        segments.push({
            text: textArea ? textArea.value : '',
            ref_audio_id: speakerSelect ? speakerSelect.value : '',
            language: languageSelect ? languageSelect.value : 'auto'
        });
    });
    
    return segments;
}

/**
 * Validate multi-speaker segments
 * @returns {Object|null} Validation result {valid: boolean, error?: string}
 */
function validateMultiSpeakerSegments() {
    const segments = getMultiSpeakerSegments();
    
    if (segments.length === 0) {
        return { valid: false, error: 'Please add at least one segment' };
    }
    
    for (let i = 0; i < segments.length; i++) {
        if (!segments[i].ref_audio_id) {
            return { valid: false, error: `Segment ${i + 1}: Please select a reference speaker` };
        }
        if (!segments[i].text.trim()) {
            return { valid: false, error: `Segment ${i + 1}: Please enter text to generate` };
        }
    }
    
    return { valid: true, segments: segments };
}

/**
 * Refresh all segment speaker dropdowns
 */
function refreshAllSegmentDropdowns() {
    document.querySelectorAll('.speaker-segment').forEach(segment => {
        const segmentId = segment.id.replace('segment-', '');
        populateSegmentSpeakerDropdown(segmentId);
    });
}
