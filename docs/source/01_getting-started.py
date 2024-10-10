import marimo

__generated_with = "0.9.4"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 1. Getting started

        There are three ways of using Djalgo: in the browser without worrying to install anything or in a Python environment for better speed and access to its AI capabilities.

        ## 1.1. In the browser

        To use Djalgo in the browser, head to a notebook supporting Pyodide, a Python interpreter that runs in the browser. Pyodide doesn't run in the cloud, it's really your browser that is compiling code. Head to [marimo.new](https://marimo.new) - [Jupyter lite](https://jupyter.org/try-jupyter/lab/) isn't supported yet ([issue](https://github.com/essicolo/djalgo/issues/4)). If you are new to programming, I highly recommend Marimo as it takes care of installing packages on import. Then use Djalgo as you intend, maybe following the tutorials with copy and paste. Since Pyodide might shut down everything when you close the tab, take care to save-as or copy-paste your code to recover your work later on.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 1.2. In a Python environment

        My recommendation to start with Python is to [install `uv`](https://docs.astral.sh/uv/getting-started/installation/). It allows managing your Python workflow like a boss. Once you have `uv` installed, use it to download the latest version of Python and create a new environment. In a terminal,

        ```
        uv python install 3.12
        cd my_project
        uv python pin cpython@3.12
        uv init .
        uv venv
        ```

        Explained:

        - `uv python install 3.12` downloads and installs the latest version of Python 3.12
        - `cd my_project` changes the directory to a new folder named `my_project`
        - `uv python pin cpython@3.12` tells `uv` to use Python 3.12 for this folder
        - `uv init .` initializes the current folder as a Python project
        - `uv venv` creates a virtual environment in the folder
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Then, activate the virtual environment and install Djalgo. In the terminal, for Linux and MacOS:

        ```
        source .venv/bin/activate
        ```

        For Windows:

        ``` 
        .venv\Scripts\activate
        ```

        Then, install Djalgo:

        ```
        uv add djalgo
        ```

        You might need to install some other packages to use Djalgo to its full potential:

        - `uv add marimo` or `uv add jupyterlab` to interact with your code in a notebook
        - Musecore and `uv add music21` to use musical Python objects
        - `uv add pretty-midi` to fine-tune and export your music to midi objects
        - `uv add scamp` to play music with sound fonts right from your notebook
        - Djalgo's AI being based on PyTorch, so you will need to `uv add torch` to train an AI on your music
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Although Djalgo comes with its a music player and a score renderer, you might want to install Musecore if you really need it. I'm showing how here, but you mostly won't need it. Once Musescore is installed, you will have to tell Python where it is. 

        - Windows: there are several ways to install MuseScore, explore to find the correct path if it doesn't work.
        ```
        from music21 import environment
        environment.set("musescoreDirectPNGPath", "C:\\Program Files\\MuseScore 4\\bin\\MuseScore.exe")
        ```

        - macOS
        ```
        from music21 import environment
        environment.set("musescoreDirectPNGPath", "/Applications/MuseScore 4.app/Contents/MacOS/mscore")
        ```

        - Linux: the command will likely work if MuseScore is installed on the system, not from a flatpak, a snap or an appimage.
        ```
        from music21 import environment
        environment.set("musescoreDirectPNGPath", "/usr/bin/musecore")
        ```
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 1.3 Leveraging Cloud Computing

        Although not my favorite approach, platforms like [Deepnote](https://deepnote.com/) and [Google Colab](https://colab.research.google.com/) can run your code in the cloud from a notebook-style interface. Bear in mind, though, that free plans generally come with service-specific limitations. While processor and RAM limitations won't hinder simple music composition methods, more advanced techniques like [machine learning-based ones](api.html#djalgo.djai.DjFlow) could experience a slower workflow. In any case, opening your pockets to a service you love can be rewarding, or course for your service provider, but also for you. To install Djalgo in a cloud environment, just run

        ```
        !pip install djalgo
        !apt update
        !apt install musescore -y
        ```

        as you would do locally on a Linux (Debian) system.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Djalgo is designed to generate musical pieces as generic Python objects.

        - A note is defined as a tuple of (midi pitch, duration time, time offset from the start). A rest is a note with a pitch defined as `None`. A rhythm is the same thing of a note, but without pitch, i.e. a (duration, offset) tuple.
        - A track is a list of notes.
        - A piece is a list of tracks.

        Such objects can be converted to several music packages in Python, like Music21, Pretty-Midi, Mido, SCAMP and ABC notation.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 1.4 Starting a (jam) session

        To start with djalgo, install it then lauch your session by importing the package. The alias `dj` will be used through the documentation.
        """
    )
    return


@app.cell
def __():
    import djalgo as dj
    return (dj,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Djalgo offera a range of functionalities designed for music composers and enthusiasts. Here’s a snapshot of what Djalgo brings to the table:

        - A **music player** in `player.py` to listen and record to your creations.
        - A **score renderer** in `score.py` to visualize your music in a score.
        - **Analysis**: Located in `analysis.py`, discover a suite of indices for dissecting tracks—whether it’s pitches, durations, or offsets. These metrics serve not just for analysis but also as benchmarks for the evolutionary algorithms found in `genetic.py`.
        - **Conversion**: `conversion.py` is your gateway to integrating Djalgo with popular music packages. Transform notes and compositions into formats compatible with Music21 for notation, Pretty-Midi for MIDI refinements, and SCAMP for sound production. Installing these packages is a prerequisite for conversion.
        - **Fractals**: `fractal.py` delves into the beauty of mathematics, extracting music from the intricate patterns of cellular automata and Mandelbrot fractals.
        - **Genetic Algorithms**: Use `genetic.py` to evolve your music, steering it towards specific analytical targets defined in `analysis.py`.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        - **Harmony**: `harmony.py` equips you with tools to enrich compositions with scales, voicing, and ornamental touches.
        - **Loop Visualization**: `loop.py` helps visualize musical loops with radar plots, offering a new dimension to beat creation.
        - **Minimalism**: Explore minimalist techniques in `minimalism.py`, from additive and subtractive processes to Arvo Pärt’s tintinnabuli, and craft music with a minimalist ethos.
        - **Rhythm**: The `Rhythm` class in `rhythm.py` is designed for crafting and experimenting with complex rhythmic structures.
        - **Utilities**: `utils.py` provides essential tools for fine-tuning: repair, tune, and quantize your compositions to perfection.
        - **Random and Kernel Walks**: In `walk.py`, let music wander through algorithmic paths, guided by random and kernel-induced walks.
        - **Machine learning**: Djai is a work in progress, and does not behave as expected. `djai.py` is  aimed at using machine learning to generate new music pieces, all powered by TensorFlow. Djai was designed to learn from any MIDI file, but if *art* your aim, you'd better create a machine that learns from your own compositions.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
