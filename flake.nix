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
          config.allowUnsupportedSystem = true;
        };

        isDarwin = pkgs.stdenv.isDarwin;

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

        # Common packages for all platforms
        commonPackages = with pkgs; [
          pythonEnv # Python 3.11 from nixpkgs (NixOS compatible)
          uv # from latest nixpkgs
          ffmpeg # from latest nixpkgs
          sox # for audio processing
          stdenv.cc # for building extensions
          stdenv.cc.cc.lib
        ];

        # Linux-specific packages (CUDA support + vlc which depends on alsa-lib)
        linuxPackages = with pkgs; [
          glibc
          vlc # vlc depends on alsa-lib which is Linux-only
        ];

        devShell = pkgs.mkShell {
          buildInputs = commonPackages ++ (if isDarwin then [ ] else linuxPackages);

          shellHook = ''
            export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
            export UV_PYTHON="${pythonEnv.interpreter}"
            export PATH="${pkgs.sox}/bin:$PATH"
            echo "Python: $(python --version)"
          '' + (if isDarwin then "" else ''
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
            export CUDA_PATH="/run/opengl-driver"
          '');
        };

      in { devShells.default = devShell; });
}
