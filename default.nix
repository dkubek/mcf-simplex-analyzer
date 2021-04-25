{ pkgs ? import <nixpkgs> {} }:

let
  poetry2nix = pkgs.poetry2nix;
  customOverrides = import ./overrides.nix { inherit pkgs; };

  mcf-simplex-analyzer = poetry2nix.mkPoetryPackages {
    projectDir = ./.;

    python = pkgs.python38;

    overrides = poetry2nix.overrides.withDefaults customOverrides;
  };
in
  mcf-simplex-analyzer
