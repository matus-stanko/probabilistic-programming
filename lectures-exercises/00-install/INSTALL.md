# Installation instrucions for PyMC

These notes are deliberately kept outside the notebooks, as you might be trying
to install things before you can read the notebook.

Tested on Arch Linux. It should work on macOS and most Linux distributions with
only small tweaks; Windows should also work by using Powershell.

## What is `uv` (and why should you care)

[Yet another package manager for python](https://xkcd.com/927/). But this one
is actually better uv is an emerging package & project manager. An alternative
to pip, poetry and conda. It promises faster installs than pip and better
dependency handling (a sore spot in the Python ecosystem), and it keeps
environments self-contained inside the project folder.

Follow the official installation at the uv docs found here:
https://docs.astral.sh/uv/getting-started/installation/

A few points beforehand:
- We recommend the direct install method (Standalone Installer)
- If you absolutely can't do it, install uv on a pip instance

## create course project

The uv manager keeps everything inside the project folder. Dependencies are
declared in `pyproject.toml` at the project root, and the environment is
created under `.venv` (so nothing global gets wrecked).

To install the packages necessary do the following:

1. Open a terminal in the course folder (the one that contains pyproject.toml).
2. Run `uv sync`

This command reads pyproject.toml, creates a virtual environment with the
specified Python version, and installs compatible packages. You should see a
.venv folder and a uv.lock file appear if everything worked.

to add packages later, run `uv add <package-name>` in the same directory. This
updates pyproject.toml and installs the package into the project environment.
If you edit pyproject.toml by hand, run `uv sync` afterward so the `.venv` and
`uv.lock` reflect the changes.

## Using uv with jupyter
Good news: the pyproject.toml for the course already includes ipython and
jupyterlab, so you should be covered.

### From an IDE (VScode, etc.)
If your IDE detects virtual environments, point the notebook kernel to the
`.venv` in the project. When opening a notebook, choose the `.venv` kernel and
enjoy.

### Run Jupterlab through uv

If you want to run the JupyterLab instance that's tied to the uv environment, use:
```
uv run --with jupyter jupyter lab
```
Then, in the notebook kernel selector, choose the kernel that corresponds to
the .venv (often shown as Python 3 (ipykernel) or similar).

### Create a standalone kernel from the `.venv`

If you prefer using a system-wide JupyterLab or want the project environment to
appear as a kernel option everywhere, install an ipykernel from the uv
environment:
 ```
 uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=prpro-2026
```

That will add a kernel named prpro-2026 (with a readable display name) to your
Jupyter kernels list.

## Deleting the environment

This is as easy as removing the `.venv` folder and `uv.lock` files

- **uv sounds too good to be true. What's the catch?**
  
  uv focuses on Python dependencies only. If your project needs system
  libraries (CUDA, certain drivers, or other non-Python runtime dependencies),
  you can make them work in uv, but with some work. It also doesn’t manage
  other languages (R, Julia, etc.). If you need multi-language kernel support,
  uv alone won’t solve that.

- **I ran into a problem. Now what?**
  
  Your TA will be happy to help you fix any issues. If it's outside the study
  sessions, post the problem on the course forum so your classmates can benefit
  from the fix. Be sure to include:

  - The error message
  - Your OS and Python version
  - A short list of steps you ran (e.g. uv sync output)

### Troubleshooting quicktips

- If uv sync fails, check the uv.lock file for clues and consider deleting
  .venv and trying again.

- If Jupyter doesn't show the .venv kernel, make sure you ran the ipykernel
  install command (see above).

- If a package seems missing in the notebook but works in the terminal,
  double-check that both are using the same .venv interpreter.
