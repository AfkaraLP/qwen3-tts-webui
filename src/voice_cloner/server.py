from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import uuid
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Union
import logging
import tempfile

from voice_cloner import VoiceCloner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Qwen TTS WebUI API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global voice cloner instance
voice_cloner = None

# Directory setup
REFERENCE_DIR = Path("reference_audios")
OUTPUT_DIR = Path("output")
REFERENCE_METADATA_FILE = Path("reference_metadata.json")
GENERATED_METADATA_FILE = Path("generated_metadata.json")

# Ensure directories exist
REFERENCE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def safe_delete_file(file_path: Union[str, Path]) -> None:
    """Safely delete a file if it exists"""
    try:
        Path(file_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Failed to delete file {file_path}: {e}")


# Initialize metadata files
if not REFERENCE_METADATA_FILE.exists():
    with open(REFERENCE_METADATA_FILE, "w") as f:
        json.dump({}, f)

if not GENERATED_METADATA_FILE.exists():
    with open(GENERATED_METADATA_FILE, "w") as f:
        json.dump({}, f)


class CloneRequest(BaseModel):
    text: str
    ref_audio_id: Optional[str] = None
    ref_text: Optional[str] = None
    language: str = "auto"


class SpeakerSegment(BaseModel):
    """A single speaker segment for multi-speaker generation"""

    text: str
    ref_audio_id: str
    ref_text: Optional[str] = None
    language: str = "auto"


class MultiSpeakerCloneRequest(BaseModel):
    """Request for multi-speaker sequential generation"""

    segments: List[SpeakerSegment]


class YouTubeCloneRequest(BaseModel):
    text: str
    youtube_url: str
    ref_text: Optional[str] = None
    language: str = "auto"
    name: Optional[str] = None


class RenameRequest(BaseModel):
    name: str


class CloneResponse(BaseModel):
    task_id: str
    status: str
    output_path: Optional[str] = None
    message: str
    estimated_time: Optional[int] = None  # Estimated time in seconds


class ReferenceAudio(BaseModel):
    id: str
    filename: str
    original_name: str
    name: Optional[str] = None
    ref_text: Optional[str] = None
    created_at: str


class GeneratedAudio(BaseModel):
    id: str
    filename: str
    ref_audio_id: str
    ref_audio_name: str
    generated_text: str
    created_at: str


# In-memory task tracking
tasks = {}
cancelled_tasks = set()


def reindex_generated_audios():
    """Reindex generated audios from disk and metadata on startup.

    This ensures that after a server restart, previously generated audios
    are still accessible via the /tasks/{task_id}/audio endpoint.
    """
    global tasks

    generated_metadata = load_generated_metadata()
    reindexed_count = 0
    removed_count = 0
    metadata_updated = False

    # First, reindex from metadata
    for task_id, data in list(generated_metadata.items()):
        file_path = OUTPUT_DIR / data["filename"]
        if file_path.exists():
            # Repopulate the tasks dictionary with completed task info
            tasks[task_id] = {
                "status": "completed",
                "progress": 100,
                "output_path": str(file_path),
                "ref_audio_id": data.get("ref_audio_id", ""),
            }
            reindexed_count += 1
        else:
            # File no longer exists, remove from metadata
            del generated_metadata[task_id]
            metadata_updated = True
            removed_count += 1

    # Also scan the output directory for any orphaned files not in metadata
    # (e.g., if metadata was corrupted or manually edited)
    for audio_file in OUTPUT_DIR.glob("cloned_*.wav"):
        # Extract task_id from filename (format: cloned_{task_id}.wav)
        filename = audio_file.name
        if filename.startswith("cloned_") and filename.endswith(".wav"):
            task_id = filename[7:-4]  # Remove "cloned_" prefix and ".wav" suffix

            if task_id not in generated_metadata:
                # Found orphaned file, add to metadata and tasks
                generated_metadata[task_id] = {
                    "filename": filename,
                    "ref_audio_id": "",
                    "ref_audio_name": "Unknown",
                    "generated_text": "",
                    "created_at": str(audio_file.stat().st_ctime),
                }
                tasks[task_id] = {
                    "status": "completed",
                    "progress": 100,
                    "output_path": str(audio_file),
                    "ref_audio_id": "",
                }
                metadata_updated = True
                reindexed_count += 1
                logger.info(f"Recovered orphaned audio file: {filename}")

    # Save updated metadata if changes were made
    if metadata_updated:
        save_generated_metadata(generated_metadata)

    if reindexed_count > 0 or removed_count > 0:
        logger.info(
            f"Reindexed {reindexed_count} generated audio(s), "
            f"removed {removed_count} stale metadata entries"
        )


def initialize_voice_cloner():
    global voice_cloner
    if voice_cloner is None:
        logger.info("Initializing VoiceCloner...")
        try:
            voice_cloner = VoiceCloner()
            logger.info("VoiceCloner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VoiceCloner: {e}")
            raise


def load_reference_metadata():
    try:
        with open(REFERENCE_METADATA_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading reference metadata: {e}")
        return {}


def save_reference_metadata(metadata):
    try:
        with open(REFERENCE_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving reference metadata: {e}")


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def find_duplicate_by_file_hash(file_hash: str, metadata: dict) -> Optional[str]:
    """Find existing reference with same file hash"""
    for audio_id, data in metadata.items():
        if "file_hash" in data and data["file_hash"] == file_hash:
            return audio_id
    return None


def load_generated_metadata():
    try:
        with open(GENERATED_METADATA_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading generated metadata: {e}")
        return {}


def save_generated_metadata(metadata):
    try:
        with open(GENERATED_METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving generated metadata: {e}")


def estimate_generation_time(text_length: int) -> int:
    """Estimate generation time based on text length (rough estimate: ~3 seconds per 10 characters)"""
    return max(30, (text_length // 10) * 3)  # Minimum 30 seconds


def download_youtube_audio(youtube_url: str) -> str:
    """Download audio from YouTube URL and return path to downloaded file"""
    temp_dir = tempfile.mkdtemp()

    try:
        # Use subprocess to call yt-dlp directly to avoid typing issues
        import subprocess

        cmd = [
            "yt-dlp",
            "--format",
            "bestaudio/best",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--audio-quality",
            "192",
            "--output",
            os.path.join(temp_dir, "audio.%(ext)s"),
            "--quiet",
            "--no-warnings",
            youtube_url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"yt-dlp failed: {result.stderr}")

        # Find the downloaded audio file
        for file in os.listdir(temp_dir):
            if file.startswith("audio") and file.endswith(".wav"):
                return os.path.join(temp_dir, file)

        raise Exception("Audio file not found after download")

    except Exception as e:
        # Clean up temp directory on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"Failed to download YouTube audio: {e}")


def trim_audio_to_max_length(audio_path: str, max_duration: int = 60) -> str:
    """Trim audio to maximum duration (in seconds)"""
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_path)
        if len(audio) > max_duration * 1000:  # Convert to milliseconds
            audio = audio[: max_duration * 1000]
            # Export trimmed audio
            trimmed_path = audio_path.replace(".wav", "_trimmed.wav")
            audio.export(trimmed_path, format="wav")
            return trimmed_path
        return audio_path
    except Exception as e:
        logger.warning(f"Failed to trim audio {audio_path}: {e}")
        return audio_path


def get_supported_languages():
    """Return list of supported languages"""
    return [
        "auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    ]


@app.on_event("startup")
async def startup_event():
    # Reindex generated audios first (before voice cloner init, as it's faster)
    reindex_generated_audios()
    # Initialize the voice cloner model
    initialize_voice_cloner()


@app.get("/", response_class=HTMLResponse)
async def get_web_ui():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/qwentts.ico")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "voice_cloner_loaded": voice_cloner is not None}


@app.post("/upload-reference", response_model=ReferenceAudio)
async def upload_reference_audio(
    file: UploadFile = File(...), ref_text: Optional[str] = Form(None)
):
    if not voice_cloner:
        raise HTTPException(status_code=503, detail="Voice cloner not initialized")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Save file temporarily to compute hash
    temp_path = REFERENCE_DIR / f"temp_upload_{uuid.uuid4()}.tmp"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Trim audio to max 1 minute before processing
    trimmed_path = trim_audio_to_max_length(str(temp_path), 60)

    # Compute file hash on trimmed audio
    file_hash = compute_file_hash(Path(trimmed_path))

    # Check for duplicate (using trimmed audio hash)
    metadata = load_reference_metadata()
    existing_audio_id = find_duplicate_by_file_hash(file_hash, metadata)

    if existing_audio_id:
        # Clean up temp files
        temp_path.unlink()
        if trimmed_path != temp_path:
            Path(trimmed_path).unlink()

        # Return existing reference
        existing_data = metadata[existing_audio_id]
        return ReferenceAudio(
            id=existing_audio_id,
            filename=existing_data["filename"],
            original_name=existing_data["original_name"],
            ref_text=existing_data.get("ref_text"),
            created_at=existing_data["created_at"],
        )

    # Generate unique ID for new file
    audio_id = str(uuid.uuid4())
    file_extension = Path(file.filename if file.filename else "audio").suffix or ".wav"
    saved_filename = f"{audio_id}{file_extension}"
    saved_path = REFERENCE_DIR / saved_filename

    # Move trimmed file to final location
    try:
        shutil.move(trimmed_path, str(saved_path))
        # Clean up original temp file
        temp_path.unlink()
    except Exception as e:
        # Clean up temp files on error
        temp_path.unlink()
        if trimmed_path != temp_path:
            Path(trimmed_path).unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Auto-transcribe if ref_text not provided (after trimming)
    if not ref_text:
        try:
            ref_text = voice_cloner.transcribe_audio(str(saved_path))
        except Exception as e:
            logger.warning(f"Failed to auto-transcribe {saved_path}: {e}")

    # Save metadata with file hash
    metadata[audio_id] = {
        "filename": saved_filename,
        "original_name": file.filename if file.filename else "unknown",
        "ref_text": ref_text,
        "file_hash": file_hash,
        "created_at": str(Path(saved_path).stat().st_ctime),
    }
    save_reference_metadata(metadata)

    return ReferenceAudio(
        id=audio_id,
        filename=saved_filename,
        original_name=file.filename if file.filename else "unknown",
        ref_text=ref_text,
        created_at=metadata[audio_id]["created_at"],
    )


@app.get("/references", response_model=List[ReferenceAudio])
async def get_reference_audios():
    metadata = load_reference_metadata()
    references = []
    metadata_updated = False

    for audio_id, data in metadata.items():
        # Check if file still exists
        file_path = REFERENCE_DIR / data["filename"]
        if file_path.exists():
            # Add file hash if not present (backward compatibility)
            if "file_hash" not in data:
                try:
                    data["file_hash"] = compute_file_hash(file_path)
                    metadata_updated = True
                except Exception as e:
                    logger.warning(f"Failed to compute hash for {file_path}: {e}")

            references.append(
                ReferenceAudio(
                    id=audio_id,
                    filename=data["filename"],
                    original_name=data["original_name"],
                    name=data.get("name"),
                    ref_text=data.get("ref_text"),
                    created_at=data["created_at"],
                )
            )

    # Save metadata if we updated it with file hashes
    if metadata_updated:
        save_reference_metadata(metadata)

    return references


@app.get("/references/{audio_id}/audio")
async def get_reference_audio(audio_id: str):
    metadata = load_reference_metadata()
    if audio_id not in metadata:
        raise HTTPException(status_code=404, detail="Reference audio not found")

    file_path = REFERENCE_DIR / metadata[audio_id]["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        file_path, media_type="audio/wav", filename=metadata[audio_id]["original_name"]
    )


@app.delete("/references/{audio_id}")
async def delete_reference_audio(audio_id: str):
    metadata = load_reference_metadata()
    if audio_id not in metadata:
        raise HTTPException(status_code=404, detail="Reference audio not found")

    # Delete file
    file_path = REFERENCE_DIR / metadata[audio_id]["filename"]
    if file_path.exists():
        file_path.unlink()

    # Remove from metadata
    del metadata[audio_id]
    save_reference_metadata(metadata)

    return {"message": "Reference audio deleted successfully"}


@app.put("/references/{audio_id}/name", response_model=dict)
async def rename_reference_audio(audio_id: str, request: RenameRequest):
    metadata = load_reference_metadata()
    if audio_id not in metadata:
        raise HTTPException(status_code=404, detail="Reference audio not found")

    # Update the name in metadata
    metadata[audio_id]["name"] = request.name
    save_reference_metadata(metadata)

    return {"message": "Reference audio renamed successfully", "name": request.name}


@app.post("/clone", response_model=CloneResponse)
async def clone_voice(background_tasks: BackgroundTasks, request: CloneRequest):
    if not voice_cloner:
        raise HTTPException(status_code=503, detail="Voice cloner not initialized")

    # Validate request
    if not request.ref_audio_id:
        raise HTTPException(status_code=400, detail="Reference audio ID is required")

    # Get reference audio metadata
    metadata = load_reference_metadata()
    if request.ref_audio_id not in metadata:
        raise HTTPException(status_code=404, detail="Reference audio not found")

    ref_data = metadata[request.ref_audio_id]
    ref_audio_path = REFERENCE_DIR / ref_data["filename"]

    if not ref_audio_path.exists():
        raise HTTPException(status_code=404, detail="Reference audio file not found")

    # Use provided ref_text or stored one
    ref_text = request.ref_text or ref_data.get("ref_text")
    if not ref_text:
        raise HTTPException(status_code=400, detail="Reference text is required")

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Start background task
    background_tasks.add_task(
        process_cloning_task,
        task_id,
        request.text,
        str(ref_audio_path),
        ref_text,
        request.language,
    )

    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "ref_audio_id": request.ref_audio_id,
    }

    return CloneResponse(
        task_id=task_id, status="processing", message="Voice cloning started"
    )


@app.post("/clone-with-youtube", response_model=CloneResponse)
async def clone_voice_with_youtube(
    background_tasks: BackgroundTasks, request: YouTubeCloneRequest
):
    if not voice_cloner:
        raise HTTPException(status_code=503, detail="Voice cloner not initialized")

    # Validate YouTube URL
    if not request.youtube_url:
        raise HTTPException(status_code=400, detail="YouTube URL is required")

    if (
        "youtube.com" not in request.youtube_url
        and "youtu.be" not in request.youtube_url
    ):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Download audio from YouTube
    try:
        youtube_audio_path = download_youtube_audio(request.youtube_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Trim audio to max 1 minute
    trimmed_audio_path = trim_audio_to_max_length(youtube_audio_path, 60)

    # Auto-transcribe if ref_text not provided
    ref_text = request.ref_text
    if not ref_text:
        try:
            ref_text = voice_cloner.transcribe_audio(trimmed_audio_path)
        except Exception as e:
            logger.warning(f"Failed to auto-transcribe YouTube audio: {e}")

    # Save as reference audio
    audio_id = str(uuid.uuid4())
    saved_filename = f"{audio_id}.wav"
    saved_path = REFERENCE_DIR / saved_filename

    try:
        shutil.move(trimmed_audio_path, str(saved_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Save metadata
    metadata = load_reference_metadata()
    display_name = (
        request.name if request.name else f"YouTube Audio - {request.youtube_url}"
    )
    metadata[audio_id] = {
        "filename": saved_filename,
        "original_name": f"YouTube Audio - {request.youtube_url}",
        "name": display_name,
        "ref_text": ref_text,
        "file_hash": compute_file_hash(saved_path),
        "created_at": str(Path(saved_path).stat().st_ctime),
    }
    save_reference_metadata(metadata)

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Start background task
    background_tasks.add_task(
        process_cloning_task,
        task_id,
        request.text,
        str(saved_path),
        ref_text or "",
        request.language,
    )

    tasks[task_id] = {"status": "processing", "progress": 0, "ref_audio_id": audio_id}

    return CloneResponse(
        task_id=task_id,
        status="processing",
        message="Voice cloning started with YouTube reference audio",
    )


@app.post("/clone-with-upload", response_model=CloneResponse)
async def clone_voice_with_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text: str = Form(...),
    ref_text: Optional[str] = Form(None),
    language: str = Form("auto"),
):
    if not voice_cloner:
        raise HTTPException(status_code=503, detail="Voice cloner not initialized")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Save uploaded file temporarily
    temp_path = REFERENCE_DIR / f"temp_upload_{uuid.uuid4()}.tmp"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Trim audio to max 1 minute before processing
    trimmed_path = trim_audio_to_max_length(str(temp_path), 60)

    # Compute file hash on trimmed audio
    file_hash = compute_file_hash(Path(trimmed_path))

    # Check for duplicate (using trimmed audio hash)
    metadata = load_reference_metadata()
    existing_audio_id = find_duplicate_by_file_hash(file_hash, metadata)

    ref_audio_path = None
    ref_audio_id_to_use = None

    if existing_audio_id:
        # Use existing file
        existing_data = metadata[existing_audio_id]
        ref_audio_path = REFERENCE_DIR / existing_data["filename"]
        ref_audio_id_to_use = existing_audio_id

        # Clean up temp files
        safe_delete_file(temp_path)
        if trimmed_path != str(temp_path):
            safe_delete_file(trimmed_path)

        # Use existing ref_text if not provided
        if not ref_text:
            ref_text = existing_data.get("ref_text")
    else:
        # New file - move trimmed file to final location
        audio_id = str(uuid.uuid4())
        file_extension = (
            Path(file.filename if file.filename else "audio").suffix or ".wav"
        )
        saved_filename = f"{audio_id}{file_extension}"
        ref_audio_path = REFERENCE_DIR / saved_filename
        ref_audio_id_to_use = audio_id

        try:
            shutil.move(trimmed_path, str(ref_audio_path))
            # Clean up original temp file
            safe_delete_file(temp_path)
        except Exception as e:
            # Clean up temp files on error
            safe_delete_file(temp_path)
            if trimmed_path != str(temp_path):
                safe_delete_file(trimmed_path)
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

        # Auto-transcribe if ref_text not provided (after trimming)
        if not ref_text:
            try:
                ref_text = voice_cloner.transcribe_audio(str(ref_audio_path))
            except Exception as e:
                logger.warning(f"Failed to auto-transcribe {ref_audio_path}: {e}")

        # Save metadata with file hash
        metadata[audio_id] = {
            "filename": saved_filename,
            "original_name": file.filename if file.filename else "unknown",
            "ref_text": ref_text,
            "file_hash": file_hash,
            "created_at": str(Path(ref_audio_path).stat().st_ctime),
        }
        save_reference_metadata(metadata)

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Determine ref_audio_id to use
    ref_audio_id_to_store = (
        existing_audio_id if existing_audio_id else ref_audio_id_to_use
    )

    # Start background task
    background_tasks.add_task(
        process_cloning_task,
        task_id,
        text,
        str(ref_audio_path),
        ref_text or "",  # Ensure ref_text is never None
        language,
    )

    # Store ref_audio_id in task
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "ref_audio_id": ref_audio_id_to_store,
    }

    return CloneResponse(
        task_id=task_id,
        status="processing",
        message="Voice cloning started with uploaded reference audio",
    )


@app.post("/clone-multi-speaker", response_model=CloneResponse)
async def clone_voice_multi_speaker(
    background_tasks: BackgroundTasks, request: MultiSpeakerCloneRequest
):
    """Clone voice with multiple speakers in sequence, concatenating the results."""
    if not voice_cloner:
        raise HTTPException(status_code=503, detail="Voice cloner not initialized")

    # Validate request
    if not request.segments or len(request.segments) == 0:
        raise HTTPException(status_code=400, detail="At least one segment is required")

    # Validate and prepare all segments
    metadata = load_reference_metadata()
    prepared_segments = []

    for i, segment in enumerate(request.segments):
        if not segment.ref_audio_id:
            raise HTTPException(
                status_code=400,
                detail=f"Segment {i + 1}: Reference audio ID is required",
            )

        if segment.ref_audio_id not in metadata:
            raise HTTPException(
                status_code=404, detail=f"Segment {i + 1}: Reference audio not found"
            )

        ref_data = metadata[segment.ref_audio_id]
        ref_audio_path = REFERENCE_DIR / ref_data["filename"]

        if not ref_audio_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Segment {i + 1}: Reference audio file not found",
            )

        # Use provided ref_text or stored one
        ref_text = segment.ref_text or ref_data.get("ref_text")
        if not ref_text:
            raise HTTPException(
                status_code=400, detail=f"Segment {i + 1}: Reference text is required"
            )

        if not segment.text.strip():
            raise HTTPException(
                status_code=400, detail=f"Segment {i + 1}: Text to generate is required"
            )

        prepared_segments.append(
            {
                "text": segment.text,
                "ref_audio_path": str(ref_audio_path),
                "ref_text": ref_text,
                "language": segment.language,
            }
        )

    # Generate task ID
    task_id = str(uuid.uuid4())

    # Start background task
    background_tasks.add_task(
        process_multi_speaker_cloning_task,
        task_id,
        prepared_segments,
    )

    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "ref_audio_id": "multi-speaker",
        "is_multi_speaker": True,
        "total_segments": len(prepared_segments),
        "current_segment": 0,
    }

    return CloneResponse(
        task_id=task_id,
        status="processing",
        message=f"Multi-speaker voice cloning started with {len(prepared_segments)} segments",
    )


def process_cloning_task(
    task_id: str, text: str, ref_audio_path: str, ref_text: str, language: str
):
    try:
        global voice_cloner
        if not voice_cloner:
            raise Exception("Voice cloner not initialized")

        # Check if task was cancelled
        if task_id in cancelled_tasks:
            tasks[task_id]["status"] = "cancelled"
            return

        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 25

        # Check if task was cancelled
        if task_id in cancelled_tasks:
            tasks[task_id]["status"] = "cancelled"
            return

        # Perform voice cloning
        output_filename = f"cloned_{task_id}.wav"
        output_path = OUTPUT_DIR / output_filename

        tasks[task_id]["progress"] = 50

        # Check if task was cancelled
        if task_id in cancelled_tasks:
            tasks[task_id]["status"] = "cancelled"
            return

        sr, result_path = voice_cloner.clone_voice(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            language=language,
            output_path=str(output_path),
        )

        # Check if task was cancelled after cloning
        if task_id in cancelled_tasks:
            tasks[task_id]["status"] = "cancelled"
            # Clean up output file if cancelled
            if output_path.exists():
                output_path.unlink()
            return

        tasks[task_id]["progress"] = 90

        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["output_path"] = str(output_path)

        # Save to generated metadata
        generated_metadata = load_generated_metadata()

        # Get reference audio info for name
        reference_metadata = load_reference_metadata()
        ref_audio_name = "Unknown"
        for ref_id, ref_data in reference_metadata.items():
            if ref_id == tasks[task_id].get("ref_audio_id"):
                ref_audio_name = ref_data.get("name") or ref_data.get(
                    "original_name", "Unknown"
                )
                break

        generated_metadata[task_id] = {
            "filename": output_filename,
            "ref_audio_id": tasks[task_id].get("ref_audio_id", ""),
            "ref_audio_name": ref_audio_name,
            "generated_text": text,
            "created_at": str(Path(output_path).stat().st_ctime),
        }
        save_generated_metadata(generated_metadata)

    except Exception as e:
        logger.error(f"Cloning task {task_id} failed: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)


def process_multi_speaker_cloning_task(
    task_id: str,
    segments: List[dict],
):
    """Process multi-speaker voice cloning task.

    Args:
        task_id: Unique task identifier
        segments: List of dicts with keys: text, ref_audio_path, ref_text, language
    """
    temp_audio_paths: List[str] = []
    try:
        global voice_cloner
        if not voice_cloner:
            raise Exception("Voice cloner not initialized")

        # Check if task was cancelled
        if task_id in cancelled_tasks:
            tasks[task_id]["status"] = "cancelled"
            return

        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 5

        total_segments = len(segments)

        # Process each segment
        for i, segment in enumerate(segments):
            # Check if task was cancelled
            if task_id in cancelled_tasks:
                tasks[task_id]["status"] = "cancelled"
                # Clean up temp files
                for temp_path in temp_audio_paths:
                    safe_delete_file(temp_path)
                return

            # Update progress (distribute 5-90% across segments)
            segment_progress = 5 + int((85 * i) / total_segments)
            tasks[task_id]["progress"] = segment_progress
            tasks[task_id]["current_segment"] = i + 1
            tasks[task_id]["total_segments"] = total_segments

            # Generate audio for this segment
            temp_output_path = OUTPUT_DIR / f"temp_{task_id}_segment_{i}.wav"

            sr, result_path = voice_cloner.clone_voice(
                text=segment["text"],
                ref_audio=segment["ref_audio_path"],
                ref_text=segment["ref_text"],
                language=segment["language"],
                output_path=str(temp_output_path),
            )

            temp_audio_paths.append(str(temp_output_path))

        # Check if task was cancelled before concatenation
        if task_id in cancelled_tasks:
            tasks[task_id]["status"] = "cancelled"
            for temp_path in temp_audio_paths:
                safe_delete_file(temp_path)
            return

        tasks[task_id]["progress"] = 90

        # Concatenate all audio segments
        output_filename = f"cloned_{task_id}.wav"
        output_path = OUTPUT_DIR / output_filename

        from voice_cloner import VoiceCloner as VC

        VC.concatenate_audios(temp_audio_paths, str(output_path))

        # Clean up temp files
        for temp_path in temp_audio_paths:
            safe_delete_file(temp_path)

        # Check if task was cancelled after concatenation
        if task_id in cancelled_tasks:
            tasks[task_id]["status"] = "cancelled"
            safe_delete_file(output_path)
            return

        tasks[task_id]["progress"] = 95

        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["output_path"] = str(output_path)

        # Save to generated metadata
        generated_metadata = load_generated_metadata()

        # Create combined text summary for metadata
        combined_text = " | ".join([seg["text"][:50] for seg in segments])
        if len(combined_text) > 200:
            combined_text = combined_text[:197] + "..."

        generated_metadata[task_id] = {
            "filename": output_filename,
            "ref_audio_id": "multi-speaker",
            "ref_audio_name": f"Multi-Speaker ({total_segments} segments)",
            "generated_text": combined_text,
            "created_at": str(Path(output_path).stat().st_ctime),
            "is_multi_speaker": True,
            "segment_count": total_segments,
        }
        save_generated_metadata(generated_metadata)

    except Exception as e:
        logger.error(f"Multi-speaker cloning task {task_id} failed: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        # Clean up any temp files on error
        for temp_path in temp_audio_paths:
            safe_delete_file(temp_path)


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if (
        task["status"] == "completed"
        or task["status"] == "failed"
        or task["status"] == "cancelled"
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Task cannot be cancelled, current status: {task['status']}",
        )

    # Add to cancelled tasks set
    cancelled_tasks.add(task_id)
    tasks[task_id]["status"] = "cancelled"

    return {"message": "Task cancelled successfully"}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks[task_id]


@app.get("/languages")
async def get_languages():
    """Get list of supported languages"""
    return {"languages": get_supported_languages()}


@app.get("/tasks/{task_id}/audio")
async def get_cloned_audio(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "completed" or "output_path" not in task:
        raise HTTPException(status_code=400, detail="Audio not ready")

    output_path = Path(task["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_path, media_type="audio/wav", filename=f"cloned_voice_{task_id}.wav"
    )


@app.get("/generated", response_model=List[GeneratedAudio])
async def get_generated_audios():
    metadata = load_generated_metadata()
    generated_audios = []

    for audio_id, data in metadata.items():
        # Check if file still exists
        file_path = OUTPUT_DIR / data["filename"]
        if file_path.exists():
            generated_audios.append(
                GeneratedAudio(
                    id=audio_id,
                    filename=data["filename"],
                    ref_audio_id=data["ref_audio_id"],
                    ref_audio_name=data.get("ref_audio_name", "Unknown"),
                    generated_text=data.get("generated_text", ""),
                    created_at=data.get("created_at", str(file_path.stat().st_ctime)),
                )
            )

    # Sort by creation time (newest first)
    generated_audios.sort(key=lambda x: x.created_at, reverse=True)

    return generated_audios


@app.delete("/generated/{audio_id}")
async def delete_generated_audio(audio_id: str):
    """Delete a generated audio file and its metadata."""
    metadata = load_generated_metadata()

    if audio_id not in metadata:
        raise HTTPException(status_code=404, detail="Generated audio not found")

    # Get the filename and delete the file
    filename = metadata[audio_id].get("filename")
    if filename:
        file_path = OUTPUT_DIR / filename
        if file_path.exists():
            file_path.unlink()

    # Remove from metadata
    del metadata[audio_id]
    save_generated_metadata(metadata)

    # Also remove from tasks if present
    if audio_id in tasks:
        del tasks[audio_id]

    return {"message": "Generated audio deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("VOICE_CLONER_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
