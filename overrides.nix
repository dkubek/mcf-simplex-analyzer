{ pkgs ? import <nixpkgs> {} }:
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
