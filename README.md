# Language Modeling from Scratch

This repo contains my assignment attempts and notes from **Stanford's CS336: Language Modeling from Scratch**.

- Course lectures playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_
- Course website (schedule, assignments, slides, etc.): https://stanford-cs336.github.io/spring2025/

Each assignment's folder was created by manually copying from the corresponding official repo of the assignment.
I'm not using submodules. This means if fixes are made to the official repo and I want to pull them,
I'd have to copy them over manually. Here's a [sample commit](https://github.com/habbes/stanford-cs336-llm-from-scratch/commit/19438a8471ff555cca99edaa7170d42883a10e14)
where I copied over latest changes from the official repo. When doing so, be careful
not to overwrite your own changes that you want to preserve, since some updates make
changes to files you've modified (e.g. the adapters module). Verify that you didn't break
stuff by running sanity and official tests (e.g. see list of tests for assignment 1 in [the README.md file](./assignment1-basics/README.md))

My usual workflow for updating the repo after the official assignment repo is update is:

- Download latest repo source code from GitHub as a zip folder and extract it
- Copy the files to the corresponding directory in this repo e.g.:
  - `cp -r ~/Downloads/assignment1-basics-main/* assignment1-basics`
- Look at the pending changes in VS Code to see which files have changed
- Most updates are in the write up pdf, test files, and other files that I don't manually touch
- Sometimes there are changes to files that I've touched, mostly the [adapters](./assignment1-basics/tests/adapters.py). Peruse the changes to make sure you're changes to the file are preserved in the modified file.
- Run tests to make sure I nothing breaks (most times something breaks because I didn't restore my changes to the adapters).
  - I maintain a list of sanity and official tests to run in each assignment's directory [README](./assignment1-basics/README.md).

- Assignment 1:
  - [Local folder](./assignment1-basics/)
  - [Official repository](https://github.com/stanford-cs336/assignment1-basics)

Use [uv](https://docs.astral.sh/uv/) to create a python environemtn for each project (see each project's `README.md`).

To run python script, activate the project's environment in the terminal using:

```sh
# On Powershell/Windows
.\.venv\Scripts\activate

# On macOs/Linux
source .venv/bin/activate
```

Use the environment's python interpreter in VS Code to get better intellisense support.

![alt text](python-interpreter-status-bar.png)

![alt text](select-python-interpreter-path.png)

README docs use `wget` to download datasets. If you don't have wget you can
use an alternative client or grab directly from the source URL in your browser.

If you have `curl` installed, you can use it instead, but add the `-O` flag to download
the contents to a local file and the `-L` flag to follow redirects, e.g.

```sh
curl -OL https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
```