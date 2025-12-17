# This file is for "pack-cli".
from src.main import main # type: ignore

# This is for cli commands
image_name = "my-first-function"

import click
import subprocess

@click.group(help="CLI tool to manage full development cycle of projects")
def cli():
    pass

@click.command()
def build():
    command = f"pack build \
     --builder gcr.io/buildpacks/builder:google-22 \
     --env GOOGLE_FUNCTION_SIGNATURE_TYPE=http \
     --env GOOGLE_FUNCTION_TARGET=main \
     {image_name}"
    subprocess.run(command, shell=True, check=True)

@click.command()
def run():
    command = f"docker run -it --rm -ePORT=8001 -p8001:8080 {image_name}"
    subprocess.run(command, shell=True, check=True)

cli.add_command(build)
cli.add_command(run)

if __name__ == '__main__':
    cli()