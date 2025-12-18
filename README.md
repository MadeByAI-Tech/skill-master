# SKILL-MASTER:

- [SKILL-MASTER:](#skill-master)
  - [Developer](#developer)
  - [Function](#function)
  - [Appendix](#appendix)

This repository contains a source code for skill master projects.
It is built for running in the Google Cloud Run and Cloud Run Function environment using `functions-framework`.

## Developer




The repository is developed for VScode and devcontainer.
To spin up the development environment, use <kbd>cmd + shift + P</kbd> to open command palette and run `Dev Containers: Reopen in Container`.

**Prerequisites**

Before you start devcontainer, you must provide `openai` API key as a file in `.devcontainer/python/.secrets/openai`.

```txt
sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```


**Debug**

To run debug, use <kbd>F5</kbd> to start debugging.
Make sure to select `Functions Framework` option in the `Run and Debug` sidebar (<kbd>shift + CMD + D</kbd>).

You can access the function at the following URL:

http://localhost:8000

**build and run**

To build the projet for Google Cloud Run, use the convinent CLI from `./main.py`.

```bash
$ python ./main.py
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  CLI tool to manage full development cycle of projects

Options:
  --help  Show this message and exit.

Commands:
  build
  run
```

- build: will build the docker image for Cloud Run using `pack-cli`.
- run: will run the docker image locally using `docker run`.

You can access the function at the following URL:

http://localhost:8001

## Function

The current design of the function will invoke a pipeline.
To run only one step, you can specify the step name in the get parameter `function`.

- Debug: http://localhost:8000?function=<step_name>
- Run: http://localhost:8001?function=<step_name>

## Appendix 

https://github.com/goccy/bigquery-emulator