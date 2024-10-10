import marimo

__generated_with = "0.9.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 6. Fractals

        Walks does offer an internal structure unless we force them to pass through predefined points with fitted kernels. Fractals are mathematical sets that exhibit a repeating pattern at every scale. They have not only captivated mathematicians and scientists, but have also found a fascinating application in the world of music. The concept of fractals in music revolves around the idea of using these self-similar patterns to create compositions that can vary in complexity, embodying both a sense of infinity and a coherent structure. This approach to music composition allows for the exploration of new textures, forms, and sonic landscapes, pushing the boundaries of traditional musical creativity. Fractals inspired modern composers Jessie Montgomery in [Rounds for Piano and Orchestra](https://www.youtube.com/watch?v=eMYG_w6ueUg) and Dinuk Wijeratne in [Invisible cities](https://www.youtube.com/watch?v=sAK8aqAdUCA&t=1424s). Djalgo has four types of fractals: **cellular automata** (self-organizing systems governed by simple rules in a discrete grid), **Mandelbrot** (complex and infinitely detailed fractal structure), and **logistic map** (a simple mathematical model illustrating chaos theory).
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Cellular automata

        Genius composer Iannis Xenakis might have been the first to popularize (or even to use) cellular automata in music with his orchestral work [Horos](https://www.youtube.com/watch?v=9aYsh8SRB-c) in 1986  ([Solomos, 2013](https://hal.science/hal-00770141)). This section applies a rather simple method to compose music with cellular automata, and can be summarized in three steps.

        1. Select a rule (among the 256 presented in the [Wolfram atlas](http://atlas.wolfram.com/01/01/)) and the initial state, then draw the cells.
        2. Select a strip along the sequence dimension.
        3. Apply notes and durations to transform the strip to a [digital piano roll](https://en.wikipedia.org/wiki/Piano_roll#In_digital_audio_workstations).

        These steps are shown in this 1 minute 13 seconds video. The music played in this video is the one we will compose here.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""<div style="padding:56.25% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/791484908?h=8ed8a123a2&badge=0&autopause=0&player_id=0&app_id=58479/embed" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen frameborder="0" style="position:absolute;top:0;left:0;width:100%;height:100%;"></iframe></div>""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""You can create an interesting score by hand, but inductively trying all sorts of scores, or complexifying the process by hand can be labourious. This is where Djalgo can help. We can start by plotting a few cellular automata rules, with a single cell initiating the pattern at the centre of the first sequence.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""fdafda.""")
    return


@app.cell(hide_code=True)
def __(dj, plt):
    _width = 20
    _iterations = 10
    _initial_state = [0] * _width
    _initial_state[_width // 2] = 1
    _rules = [18, 22, 30, 45, 54, 60, 73, 102, 105, 110, 126, 150]
    _fig, _axs = plt.subplots(
        nrows=2, ncols=6, figsize=(8, 4)
    )  # Increased figure size for better visibility
    _axs = _axs.flatten()

    for _i, _rule in enumerate(_rules):
        _ca = dj.fractal.CellularAutomata(_rule, _width, _initial_state)
        _ca.plot(
            _iterations,
            title=f"Rule {_ca.rule_number}",
            show_axis=False,
            ax=_axs[_i],
        )

    _fig
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""My written explanation video lasted 1:13, so I wanted a piece with a length a little below that. We will create three tracks for a traditional minimal rock band with an electric guitar, an electric bass and a drum kit.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Steps 1 and 2. Generate cellular automata and select strips

        The first cellular automata will create piano rolls for the guitar and the bass. [Rule 150](http://atlas.wolfram.com/01/01/150/) seemed appropriate. A width of 200 cells will provide a good overview of the cellular automata. I inductively chose 136 steps after few iterations to obtain a musical piece a bit shorter than 1:13. For the sake of initial state, I just took a one in the middle.
        """
    )
    return


@app.cell
def __(dj):
    # Generate cells
    ca1_rule = 150
    width1 = 200
    length1 = 136
    init1 = [0] * width1
    init1[width1 // 2] = 1
    ca1 = dj.fractal.CellularAutomata(ca1_rule, width1, init1)
    ca1.plot(iterations=length1)
    return ca1, ca1_rule, init1, length1, width1


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""From this plot, I selected two strips: a guitar line playing all along the piece, and bass to jump in a little later.""")
    return


@app.cell
def __(ca1, length1):
    strips1 = [(97, 103), (85, 92)]
    ca1.plot(iterations=length1, strips=strips1)
    return (strips1,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The cellular automata process used for the guitar and the bass felt too dense with too many blank regions to generate good drumming. I tried a less dense rule set, with a random initial condition, then selected a region that allowed drums to kick in after a few beats in the piece.""")
    return


@app.cell
def __(dj):
    import random

    random.seed(123)

    ca2_rule = 18
    width2 = 200
    length2 = 136
    init2 = [0] * width2
    init2 = random.choices([0, 1], weights=[0.9, 0.1], k=width2)
    ca2 = dj.fractal.CellularAutomata(ca2_rule, width2, init2)
    # ca2.plot(iterations=length2);
    strips2 = [(110, 116)]
    ca2.plot(iterations=length2, strips=strips2)
    return ca2, ca2_rule, init2, length2, random, strips2, width2


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We can narrow our cellular automata arrays with the `extract_strip` argument.""")
    return


@app.cell
def __(ca1, length1, strips1):
    ca1.plot(
        iterations=length1,
        strips=strips1,
        title=["Guitar", "Bass"],
        extract_strip=True,
    )
    return


@app.cell
def __(ca2, length2, strips2):
    ca2.plot(
        iterations=length2, strips=strips2, title=["Drums"], extract_strip=True
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Step 3. Generate pitches

        Until now, we just selected our strips visually. The `.generate()` method creates a sequence of numbers based on the strips, mapped on the `values` argument. We must trigger `.generate()` with arguments `iterations` for the length of the sequence, `strips` as used previously, and `values`, a list of dictionaries used for mapping. 

        In our case, values are pitches. I used a C-minor scale (even for the drums, but it doesn't really matter), and durations sum to 8 to cover two 4/4 measures.
        """
    )
    return


@app.cell
def __(dj):
    guitar_p_set = [
        dj.utils.cde_to_midi(p)
        for p in ["C4", "D4", "Eb4", "F4", "G4", "Ab4", "Bb4"]
    ]
    bass_p_set = [
        dj.utils.cde_to_midi(p)
        for p in ["C3", "D3", "Eb3", "F3", "G3", "Ab3", "Bb3"]
    ]
    drum_p_set = [
        dj.utils.cde_to_midi(p)
        for p in ["G3", "Ab3", "Bb3", "C4", "D4", "Eb4", "F4"]
    ]
    return bass_p_set, drum_p_set, guitar_p_set


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The values in `.generate()` must be specified in dictionaries.""")
    return


@app.cell
def __(bass_p_set, drum_p_set, guitar_p_set):
    guitar_values = {}
    for _i, _p in enumerate(guitar_p_set):
        guitar_values[_i] = _p

    bass_values = {}
    for _i, _p in enumerate(bass_p_set):
        bass_values[_i] = _p

    drum_values = {}
    for _i, _p in enumerate(drum_p_set):
        drum_values[_i] = _p
    return bass_values, drum_values, guitar_values


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Our streams of pitches can now be generated.""")
    return


@app.cell
def __(
    bass_values,
    ca1,
    ca2,
    drum_values,
    guitar_values,
    length1,
    length2,
    strips1,
    strips2,
):
    guitar_p, bass_p = ca1.generate(length1, strips1, [guitar_values, bass_values])
    drum_p = ca2.generate(length2, strips2, [drum_values])
    return bass_p, drum_p, guitar_p


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Step 4. Map pitches to notes

        The `beatcycle` function zips cycling durations on the pitches to generate streams of notes with Djalgo's form (pitch, duration, offset).
        """
    )
    return


@app.cell
def __(bass_p, dj, drum_p, guitar_p):
    guitar_n = dj.rhythm.beatcycle(
        pitches=guitar_p, durations=[0.5, 0.5, 1, 2, 1, 1, 0.5, 1.5]
    )
    bass_n = dj.rhythm.beatcycle(
        pitches=bass_p, durations=[1, 1, 2, 0.5, 0.5, 0.5, 0.5, 2]
    )
    drum_n = dj.rhythm.beatcycle(
        pitches=drum_p, durations=[2, 1, 1, 0.5, 0.5, 1, 1, 1]
    )
    return bass_n, drum_n, guitar_n


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""To make the bass track sound right, we might prefer to play it one note at the time, not chords. We can iterate through the pitches and when comes a list, take the first element.""")
    return


@app.cell
def __(bass_n, i):
    for _i, _n in enumerate(bass_n):
        if isinstance(_n, list):
            bass_n[i] = _n[0]
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We glue the tracks together with the insert method so that they are played simultaneously.""")
    return


@app.cell
def __(bass_n, dj, drum_n, guitar_n):
    dj.player.show([guitar_n, bass_n, drum_n])
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        By exporting our piece to a midi file, we can then import it in a DAW for further processing. I imported the file `ca.mid` in a DAW called Waveform, then mapped each track to Komplete start virtual instruments. Both Waveform and Komplete start can be used without cost and are covered by many tutorials online.

        <iframe width="100%" height="200" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1429599889&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_rateateeposts=false&show_teaser=true&visual=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/user-512016957-418252282" title="motife" target="_blank" style="color: #cccccc; text-decoration: none;">motife</a> · <a href="https://soundcloud.com/user-512016957-418252282/t-cells-vogue" title="T-Cells vogue" target="_blank" style="color: #cccccc; text-decoration: none;">T-Cells vogue</a></div>

        Because cellular automata generate repeating patterns, they perform nicely for rhythmic parts. However, they will fall short for melodies.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Rotating CA

        Another way of transforming cellular automata to music is to rotate the strips. We will use drum CA to create drums. In the following code block, I used three different rules initiated randomly for kick, snare and hit-hat.
        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __(dj, plt, random):
    random.seed(123)

    instruments = ["kick", "snare", "hat"]
    ca_drum_rule = [30, 54, 150]
    drum_width = 12
    drum_length = 4
    drum_init = [0] * drum_width

    ca_drum = {}
    _fig, _axs = plt.subplots(nrows=1, ncols=3, figsize=(6, 6))
    _axs = _axs.flatten()
    for _i in range(len(ca_drum_rule)):
        drum_init = random.choices([0, 1], weights=[0.75, 0.25], k=drum_width)
        ca_drum[instruments[_i]] = dj.fractal.CellularAutomata(
            ca_drum_rule[_i], drum_width, drum_init
        )
        ca_drum[instruments[_i]].plot(
            iterations=drum_length, ax=_axs[_i], title=instruments[_i]
        )

    _fig
    return (
        ca_drum,
        ca_drum_rule,
        drum_init,
        drum_length,
        drum_width,
        instruments,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Now instead of extracting a strip as we have done before, we can flatten the 0 and 1 array on the iteration (length) axis to generate a single line of 0 and 1 per instrument.""")
    return


@app.cell
def __(ca_drum, drum_length, np):
    drum_01 = []
    for key, value in ca_drum.items():
        drum_01i = value.generate_01(drum_length)
        drum_01i = np.array(drum_01i).flatten()
        drum_01.append(drum_01i)
        print(key, ":", drum_01i)
    return drum_01, drum_01i, key, value


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""In my instrument, a kick is pitch 36, a snare is pitch 38 and a hat is pitch 42. Each beat is of quarter length 1.""")
    return


@app.cell
def __():
    drum_01_p = [36, 38, 42]
    drum_01_d = 1
    return drum_01_d, drum_01_p


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We just have to insert notes at the right offset to create Djalgo tracks, then convert to an exportable midi format, here Pretty-midi (to be installed with `!pip install pretty-midi`).""")
    return


@app.cell
def __(dj, drum_01, drum_01_d, drum_01_p):
    tracks = []
    for _i, _drum in enumerate(drum_01):
        _current_offset = 0
        _track_i = []
        for _hit in _drum:
            if _hit == 1:
                _track_i.append((drum_01_p[_i], drum_01_d, _current_offset))
            _current_offset += drum_01_d
        tracks.append(_track_i)
    dj.player.show(tracks, tempo=240)
    return (tracks,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Mandelbrot

        You might already have seen the intriguing plot of the Mandelbrot ensemble. Djalgo implements a Mandelbrot fractal generator, which can creatively be used to generate musical patterns based on the fractal data.
        """
    )
    return


@app.cell
def __(dj):
    dj.fractal.Mandelbrot().plot(figsize=(5, 5))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""By zooming the plot, you can find unexplored regions.""")
    return


@app.cell
def __(dj, plt):
    _fig, _axs = plt.subplots(2, 2, figsize=(10, 10))
    _axs = _axs.flatten()
    mb_ranges = [
        ((-2.0, 1.0), (-1.5, 1.5)),
        ((-1.5, -1.0), (0, 0.5)),
        ((-1.2, -1.1), (0.2, 0.3)),
        ((-1.150, -1.145), (0.275, 0.280)),
    ]
    for _i, ((_xrange, _yrange), _ax) in enumerate(zip(mb_ranges, _axs)):
        _mandelbrot = dj.fractal.Mandelbrot(
            dimensions=(600, 600), max_iter=500, x_range=_xrange, y_range=_yrange
        )
        if _i < (len(mb_ranges) - 1):
            _mandelbrot.plot(ax=_ax, zoom_rect=mb_ranges[_i + 1])
        else:
            _mandelbrot.plot(ax=_ax)

    _fig
    return (mb_ranges,)


@app.cell
def __(mo):
    mo.md(r"""By scanning the Mandelbrot matrix to subtract the numbers (horizontally, vertically or diagonally), you can generate a sequence of integers. The smaller value used for length pixelates the last Mandelbrot plot to create a smaller matrix, since we need fewer values for a musical sequence.""")
    return


@app.cell
def __(dj, mb_ranges, plt):
    x_range, y_range = mb_ranges[3]
    _length = 20
    mandelbrot_object = dj.fractal.Mandelbrot(
        dimensions=_length, max_iter=100, x_range=x_range, y_range=y_range
    )
    mandelbrot_integers = mandelbrot_object.generate(method="diagonal-increasing")

    _fig, _axs = plt.subplots(1, 2, figsize=(20, 8))
    _axs = _axs.flatten()
    # heatmap
    mandelbrot_object.plot(ax=_axs[0], show_numbers=False)
    # increasing diagonal
    _axs[0].plot(x_range, y_range, color="red")

    # numbers
    _axs[1].plot(range(_length), mandelbrot_integers, "-o", color="black")
    for _i, _m in enumerate(mandelbrot_integers):
        _axs[1].text(
            _i - 0.5, _m + 0.5, str(_m), fontsize=12, verticalalignment="bottom"
        )

    _fig
    return mandelbrot_integers, mandelbrot_object, x_range, y_range


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""These numbers can be set as midi pitches, indexes on scales, etc. If you need a scale, make sure the indexes are included in the scale. You can scale them to range between index 0 and index 20.""")
    return


@app.cell
def __(dj, mandelbrot_integers):
    g_major = dj.harmony.Scale(tonic="G", mode="major").generate()[32:53]
    mandelbrot_index = [
        int(i) for i in dj.utils.scale_list(mandelbrot_integers, 0, 13)
    ]
    mandelbrot_p = [g_major[i] for i in mandelbrot_index]
    mandelbrot_n = dj.rhythm.beatcycle(
        pitches=mandelbrot_p, durations=[0.5, 0.5, 1, 2, 1]
    )
    dj.player.show(mandelbrot_n)
    return g_major, mandelbrot_index, mandelbrot_n, mandelbrot_p


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Logistic map

        The logistic map comes from the logistic growth equation, which in turn comes from the concept of population growth. Disregarding limitations from the environment, a population $🐇$ of a reproducing species will grow at a certain $r$ rate.

        $$
        🐇_{t+1} = r \times 🐇_t
        $$

        That means that population $x$ at the next step depends on population $x$ at the current step times a growth rate, expressed in proprotion of the population per step.
        """
    )
    return


@app.cell
def __(plt):
    def growth(x, r):
        return r * x  # simple exponential growth


    _initial_population = 100
    _population = [_initial_population]
    _generations = 10
    _growth_rate = 2  # population doubles each generation
    for _i in range(_generations):
        _population.append(growth(_population[_i], _growth_rate))

    plt.plot(range(_generations + 1), _population, "-o")
    return (growth,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        The growth is exponential. The more the population, the more it reproduces. But populations never grows to infinite. There are always limits to population growth. We might want to decrease the growth rate over time. Or, a better way to think of it would be to decrease the growth rate according to the population: the more the population, the less the growth rate. Let's call it the limit of the environment $limit$. This is where the logistic equation is useful.

        $$
        🐇_{t+1} = r \times 🐇_t \times \frac{(limit - 🐇_t)}{limit}
        $$

        When $🐇_t$ comes close to the $limit$, the term $\frac{(limit - 🐇_t)}{limit}$ comes close to zero.
        """
    )
    return


@app.cell
def __(plt):
    def logistic_growth(x, r, limit):
        return r * x * (limit - x) / limit  # logistic growth


    _initial_population = 100
    _population = [_initial_population]
    _generations = 10
    _growth_rate = 2
    _limit = 5000
    for _i in range(_generations):
        _population.append(logistic_growth(_population[_i], _growth_rate, _limit))

    plt.plot(range(_generations + 1), _population, "-o")
    plt.title(f"Logistic Growth, rate = {_growth_rate}, limit = {_limit}")
    return (logistic_growth,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""You might have remarked that the population never reaches 5000, but only half of it. That's because the growth rate is decreased so much that population can't reach the maximum allowed. Let's see what happens with different growth rates.""")
    return


@app.cell
def __(logistic_growth, plt):
    _growth_rates = [2, 2.5, 3, 3.25]
    _initial_population = 100
    _limit = 5000
    _generations = 50
    for _r in _growth_rates:
        _population = [_initial_population]
        for _i in range(_generations):
            _population.append(logistic_growth(_population[_i], _r, _limit))
        plt.plot(range(_generations + 1), _population, label=f"rate = {_r}")
    plt.legend()
    plt.title(f"Logistic Growth")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""When growth rate increases, an oscillating convergence to a stabilized value occurs, and this value depends on the growth rate.. If we focus our interest on this stabilized value depending on growth rates, we get a very intriguing pattern.""")
    return


@app.cell
def __(dj, np):
    _lm = dj.fractal.LogisticMap(
        rates=np.linspace(2.5, 4, 1000), iterations=1000, last_n=100
    )
    _lm.plot(figsize=(6, 4))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        An explanation of this behaviour can be found on the Veritasium YouTube channel.

        <iframe width="560" height="315" src="https://www.youtube.com/embed/ovJcsL7vyrk?si=EQoJddVSfTCDMeox" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The logistic map can be used to create notes. Here, we lower the resolution to create a list of pitches.""")
    return


@app.cell
def __(dj, np, plt):
    _length = 20
    lm = dj.fractal.LogisticMap(
        rates=np.linspace(3.5, 4, _length), iterations=1000, last_n=1
    )
    _, lm_values = lm.generate()
    plt.plot(range(_length), lm_values, "-o", color="black")
    return lm, lm_values


@app.cell
def __(dj, g_major, lm_values):
    lm_index = [int(i) for i in dj.utils.scale_list(lm_values, 0, 13)]
    lm_p = [g_major[i] for i in lm_index]
    lm_n = dj.rhythm.beatcycle(pitches=lm_p, durations=[0.5, 0.5, 1, 2, 1])
    dj.score.show(lm_n, "Logistic growth")
    return lm_index, lm_n, lm_p


@app.cell
def __():
    import marimo as mo
    import djalgo as dj
    import numpy as np
    import matplotlib.pyplot as plt
    return dj, mo, np, plt


if __name__ == "__main__":
    app.run()
