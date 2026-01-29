import click
from pathlib import Path
from .core import VoiceCloner


@click.group()
@click.pass_context
def main(ctx):
    """Voice Cloner - Easy voice cloning using Qwen TTS"""
    ctx.ensure_object(dict)


@main.command()
@click.option(
    "--text", "-t", required=True, help="Text to synthesize"
)
@click.option(
    "--ref-audio", "-r", required=True, help="Path or URL to reference audio"
)
@click.option(
    "--ref-text", "-s", help="Transcription of reference audio (optional - will auto-transcribe if not provided)"
)
@click.option(
    "--output", "-o", help="Output file path"
)
@click.option(
    "--language", "-l", default="English", help="Target language"
)
@click.option(
    "--device", "-d", default="cuda:0", help="Device to use (cuda:0, cpu, etc.)"
)
@click.option(
    "--force-transcribe", "-f", is_flag=True, help="Force re-transcription even if transcript exists"
)
def clone(text, ref_audio, ref_text, output, language, device, force_transcribe):
    """Clone voice from reference audio"""
    click.echo(f"Initializing voice cloner on {device}...")
    
    try:
        cloner = VoiceCloner(device=device)
        
        # Get or create transcript
        transcript = cloner.get_or_create_transcript(
            ref_audio=ref_audio,
            ref_text=ref_text,
            force_transcribe=force_transcribe
        )
        
        click.echo(f"Cloning voice for text: {text[:50]}...")
        
        sr, output_path = cloner.clone_voice(
            text=text,
            ref_audio=ref_audio,
            ref_text=transcript,
            language=language,
            output_path=output,
        )
        
        click.echo(f"Voice cloned successfully!")
        click.echo(f"Output: {output_path}")
        click.echo(f"Sample rate: {sr} Hz")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


@main.command()
@click.option(
    "--text", "-t", required=True, help="Text to synthesize"
)
@click.option(
    "--ref-audio", "-r", required=True, help="Path to local reference audio file"
)
@click.option(
    "--ref-text", "-s", help="Transcription of reference audio (optional - will auto-transcribe if not provided)"
)
@click.option(
    "--output-dir", "-o", default="outputs", help="Output directory"
)
@click.option(
    "--language", "-l", default="English", help="Target language"
)
@click.option(
    "--device", "-d", default="cuda:0", help="Device to use"
)
@click.option(
    "--force-transcribe", "-f", is_flag=True, help="Force re-transcription even if transcript exists"
)
def clone_from_file(text, ref_audio, ref_text, output_dir, language, device, force_transcribe):
    """Clone voice from local reference audio file"""
    click.echo(f"Initializing voice cloner on {device}...")
    
    try:
        cloner = VoiceCloner(device=device)
        
        if not Path(ref_audio).exists():
            click.echo(f"Reference audio file not found: {ref_audio}", err=True)
            raise click.Abort()
        
        # Get or create transcript
        transcript = cloner.get_or_create_transcript(
            ref_audio=ref_audio,
            ref_text=ref_text,
            force_transcribe=force_transcribe
        )
            
        click.echo(f"Cloning voice for text: {text[:50]}...")
        
        output_path = cloner.clone_voice_from_file(
            text=text,
            ref_audio_path=ref_audio,
            ref_text=transcript,
            output_dir=output_dir,
            language=language,
        )
        
        click.echo(f"Voice cloned successfully!")
        click.echo(f"Output: {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()



@main.command()
@click.option(
    "--audio", "-a", required=True, help="Path to audio file to transcribe"
)
@click.option(
    "--output", "-o", help="Output transcript file path (default: audio_file.txt)"
)
@click.option(
    "--device", "-d", default="cuda:0", help="Device to use for transcription"
)
@click.option(
    "--model", "-m", default="base", help="Whisper model size (tiny, base, small, medium, large)"
)
def transcribe(audio, output, device, model):
    """Transcribe audio file using Whisper"""
    click.echo(f"Transcribing audio with Whisper {model} model...")
    
    try:
        if not Path(audio).exists():
            click.echo(f"Audio file not found: {audio}", err=True)
            raise click.Abort()
        
        cloner = VoiceCloner(device=device, whisper_model=model)
        
        transcript = cloner.transcribe_audio(
            audio_path=audio,
            output_path=output
        )
        
        click.echo(f"Transcription:")
        click.echo(f"   {transcript}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
