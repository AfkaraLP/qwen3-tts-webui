import torch
import soundfile as sf
import whisper
import os
from qwen_tts import Qwen3TTSModel
from pathlib import Path
from typing import Optional, Tuple


def get_default_device() -> str:
    """Get the default device based on availability."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_default_dtype(device: str) -> torch.dtype:
    """Get the default dtype based on device capabilities."""
    if device.startswith("cuda"):
        return torch.bfloat16
    elif device == "mps":
        # MPS supports float16 better than bfloat16
        return torch.float16
    else:
        # CPU fallback - float32 is most compatible
        return torch.float32


class VoiceCloner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        attn_implementation: str = "sdpa",
        whisper_model: str = "base",
    ):
        """Initialize the Voice Cloner with Qwen TTS model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use (cuda:0, mps, cpu, or None for auto-detect)
            dtype: Data type for model weights (or None for auto-detect based on device)
            attn_implementation: Attention implementation
            whisper_model: Whisper model size for transcription
        """
        # Auto-detect device if not specified
        if device is None:
            device = get_default_device()

        # Auto-detect dtype if not specified
        if dtype is None:
            dtype = get_default_dtype(device)

        self.device = device
        self.dtype = dtype

        print(f"Initializing model on device: {device} with dtype: {dtype}")

        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        self.whisper_model = whisper.load_model(whisper_model)

    def clone_voice(
        self,
        text: str,
        ref_audio: str,
        ref_text: str,
        language: str = "English",
        output_path: Optional[str] = None,
    ) -> Tuple[int, str]:
        """
        Clone voice from reference audio.

        Args:
            text: Text to synthesize
            ref_audio: Path or URL to reference audio file
            ref_text: Transcription of reference audio
            language: Target language
            output_path: Output file path (optional)

        Returns:
            Tuple of (sample_rate, output_path)
        """
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

        if output_path is None:
            output_path = "output_voice_clone.wav"

        sf.write(output_path, wavs[0], sr)
        return sr, output_path

    def clone_voice_from_file(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        output_dir: str = "outputs",
        language: str = "English",
    ) -> str:
        """
        Clone voice from local reference audio file.

        Args:
            text: Text to synthesize
            ref_audio_path: Path to local reference audio file
            ref_text: Transcription of reference audio
            output_dir: Directory to save output
            language: Target language

        Returns:
            Path to generated audio file
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True)

        # Generate output filename
        ref_name = Path(ref_audio_path).stem
        text_preview = text[:30].replace(" ", "_").replace("/", "_")
        output_path = output_dir_path / f"{ref_name}_{text_preview}.wav"

        sr, final_path = self.clone_voice(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            language=language,
            output_path=str(output_path),
        )

        return final_path

    def transcribe_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file
            output_path: Path to save transcript (optional)

        Returns:
            Transcription text
        """
        print(f"Transcribing audio: {audio_path}")

        result = self.whisper_model.transcribe(audio_path)
        text_result = result.get("text", "")
        if isinstance(text_result, list):
            transcript = " ".join(str(item) for item in text_result).strip()
        else:
            transcript = str(text_result).strip()

        if output_path is None:
            # Generate default output path
            audio_path_obj = Path(audio_path)
            output_path = str(audio_path_obj.with_suffix(".txt"))

        # Save transcript to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"✅ Transcription saved to: {output_path}")
        return transcript

    def get_or_create_transcript(
        self,
        ref_audio: str,
        ref_text: Optional[str] = None,
        force_transcribe: bool = False,
    ) -> str:
        """
        Get transcript text from file or generate it from audio.

        Args:
            ref_audio: Path to reference audio file
            ref_text: Path to transcript file or transcript text
            force_transcribe: Force re-transcription even if transcript file exists

        Returns:
            Transcript text
        """
        # If ref_text is provided and exists as a file, read it
        if ref_text and Path(ref_text).exists() and not force_transcribe:
            with open(ref_text, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
                if not transcript:
                    print(f"⚠️ Transcript file exists but is empty: {ref_text}")
                return transcript

        # If ref_text is provided text directly, return it
        if ref_text and not Path(ref_text).exists():
            transcript = ref_text.strip()
            if not transcript:
                print("⚠️ Provided transcript text is empty")
            return transcript

        # Otherwise, transcribe the audio file
        audio_path = Path(ref_audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio}")

        # Validate audio file
        self._validate_audio_file(str(audio_path))

        transcript_path = audio_path.with_suffix(".txt")

        # Use existing transcript if it exists and not forced
        if transcript_path.exists() and not force_transcribe:
            with open(str(transcript_path), "r", encoding="utf-8") as f:
                transcript = f.read().strip()
                if not transcript:
                    print(f"⚠️ Existing transcript file is empty, re-transcribing...")
                else:
                    return transcript

        # Transcribe the audio
        return self.transcribe_audio(ref_audio, str(transcript_path))

    def _validate_audio_file(self, audio_path: str) -> None:
        """
        Validate that the audio file is accessible and not corrupted.

        Args:
            audio_path: Path to audio file

        Raises:
            ValueError: If audio file is invalid
        """
        try:
            import soundfile as sf

            # Try to read the audio file metadata
            with sf.SoundFile(audio_path) as f:
                if f.frames <= 0:
                    raise ValueError(f"Audio file has no frames: {audio_path}")
                if f.samplerate <= 0:
                    raise ValueError(
                        f"Audio file has invalid sample rate: {audio_path}"
                    )
                print(
                    f"✅ Audio file valid: {audio_path} ({f.frames} frames, {f.samplerate} Hz)"
                )
        except Exception as e:
            raise ValueError(f"Invalid audio file {audio_path}: {str(e)}")
