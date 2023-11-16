# Introduction

This is the documentation for the `aiida-opsp` plugin. 
The plugin runs optimization on the pseudopoetential generation to find the optimal parameters for the pseudopotential generation.

The plugin contains two processes:

* The pseudopotential generation process
* The optimization process

## Quick start

We provide a docker image to setup working environment without installing the plugin and AiiDA.

First, you need to [install Docker on your workstation or laptop](https://docs.docker.com/get-docker/).

If you are using Linux or MacOS, you can run the following command to run the workflow:

```bash
docker run -it ghcr.io/unkcpz/aiida-opsp:edge bash
```

If you are using Windows system, open the Docker desktop and search for the `aiida-opsp` image to start the container, then go to "Exec" tab to open a terminal.

The image contains the [ONCVPSP code](https://github.com/oncvpsp/oncvpsp) [^1] for pseudopotential generation, and the AiiDA plugin is installed in the container for running the workflow.
Go to the [tutorial section](tutorial.md) to learn how to run the pseudopotential generation and optimization workflows.

[^1]: Hamann, D. R. "Optimized norm-conserving Vanderbilt pseudopotentials." Physical Review B 88.8 (2013): 085117.