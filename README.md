# SKILL-MASTER:

- [SKILL-MASTER:](#skill-master)
  - [Developer](#developer)
    - [Trigger](#trigger)
  - [Function](#function)
  - [Appendix](#appendix)

This repository contains a source code for skill master projects.
It is built for running in the Google Cloud Run and Cloud Run Function environment using `functions-framework`.

## Developer

The repository is developed for VScode and devcontainer.
To spin up the development environment, use <kbd>cmd + shift + P</kbd> to open command palette and run `Dev Containers: Reopen in Container`.

**Debug**

To run debug, use <kbd>F5</kbd> to start debugging.
Make sure to select `Functions Framework` option in the `Run and Debug` sidebar (<kbd>shift + CMD + D</kbd>).

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


### Trigger

The function is triggered using HTTP request.
For development, you can use the following command to test the function:

http://localhost:8000

## Function

The current design of the function will invoke a pipeline.
To run only one step, you can specify the step name in the get parameter `function`.

http://localhost:8000?function=<step_name>

## Appendix 

https://github.com/goccy/bigquery-emulator