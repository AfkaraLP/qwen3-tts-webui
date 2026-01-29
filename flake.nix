{
  description = "Voice Cloner - Easy voice cloning using Qwen TTS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        pythonEnv = pkgs.python311.withPackages (ps:
          with ps; [
            click
            pydub
            pip
            numpy
            soundfile
            # PyTorch packages need to be added manually if using nixpkgs version
            # but we'll let uv handle them instead
          ]);

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv # Python 3.11 from nixpkgs (NixOS compatible)
            uv # from latest nixpkgs
            ffmpeg # from latest nixpkgs
            sox # for audio processing
            # Add any system dependencies needed for PyTorch
            stdenv.cc # for building extensions
            stdenv.cc.cc.lib
            glibc
            vlc
          ];

          shellHook = ''
            export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
            export UV_PYTHON="${pythonEnv.interpreter}"
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
            export CUDA_PATH="/run/opengl-driver"
            export PATH="${pkgs.sox}/bin:$PATH"
            echo "Python: $(python --version)"
          '';
        };

      in { devShells.default = devShell; });
}
