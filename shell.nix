{ pkgs ? import <nixpkgs> {} }:

let
  poetry2nix = pkgs.poetry2nix;
  customOverrides = import ./overrides.nix { inherit pkgs; };

  develEnv = poetry2nix.mkPoetryEnv {
    projectDir = ./.;

    editablePackageSources = {
      my-app = ./src;
    };

    overrides = poetry2nix.overrides.withDefaults customOverrides;
  };

  mcf-simplex-analyzer = import ./default.nix {};
in
  pkgs.mkShell {
    buildInputs = [
      pkgs.poetry
      develEnv
    ];
  }
