{ pkgs ? import <nixpkgs> {} }:

let
  poetry2nix = pkgs.poetry2nix;

  develEnv = poetry2nix.mkPoetryEnv {
    projectDir = ./.;

    editablePackageSources = {
      my-app = ./src;
    };

    overrides = poetry2nix.overrides.withDefaults (
      self: super: {

        # Override llvmlite to use LLVM 10
        llvmlite = super.llvmlite.overridePythonAttrs (
          let
            llvm = pkgs.llvm_10;
          in
            old: {
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ llvm ];

              # Disable static linking
              # https://github.com/numba/llvmlite/issues/93
              postPatch = ''
                substituteInPlace ffi/Makefile.linux --replace "-static-libstdc++" ""
                substituteInPlace llvmlite/tests/test_binding.py --replace "test_linux" "nope"
              '';

              # Set directory containing llvm-config binary
              preConfigure = ''
                export LLVM_CONFIG=${llvm}/bin/llvm-config
              '';

              passthru = old.passthru // { inherit llvm; };
            }
        );
      }
    );
  };
in

pkgs.mkShell {

  buildInputs = [
    pkgs.poetry
    develEnv
  ];
}
