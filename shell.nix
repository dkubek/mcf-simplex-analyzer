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

 customVim = pkgs.vimHugeX.override {
   python = pkgs.python3.withPackages(ps: [
     ps.python-language-server
     ps.pyls-mypy ps.pyls-isort ps.pyls-black ps.jedi
     ps.pylint
     ps.flake8
   ]);
 };

  mcf-simplex-analyzer = import ./default.nix {};
in
  pkgs.mkShell {
    buildInputs = [
      pkgs.poetry
      develEnv
      customVim
    ];
  }
