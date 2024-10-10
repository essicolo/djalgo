import marimo

__generated_with = "0.9.4"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# 4. Minimalism""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The Djalgo package includes well-known minimalistic mathematical operations on notes to create no less rich musical scores. We will cover and merge two kinds of minimalistic operations: deterministic, where the outputs are consistently determined by the inputs, and stochastic, where we induce randomness. Of course, deterministic composition can be mixed with randomness. We would then refer to generative composition. In this section, I explain how to compose minimalistic, but powerful music with Djalgo.""")
    return


@app.cell
def __():
    import djalgo as dj
    import itertools
    import random
    return dj, itertools, random


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Isorhythms

        The isorhythm consist in mapping durations to pitches. The process can be done by [zipping](https://docs.python.org/3/library/functions.html#zip) together lists of pitches and rhythms, but the [isorhythm](api.html#djalgo.rhythm.isorhythm) function keeps the sequence running as long as the end of pitches and duration coincide, then the offsets will are adjusted to durations.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Example""")
    return


@app.cell
def __(dj):
    pitches_cmajor = dj.harmony.Scale(tonic="C", mode="major").generate()[35:43]
    _durations = [1] * 8
    solfege = dj.rhythm.isorhythm(pitches=pitches_cmajor, durations=_durations)
    dj.score.show(solfege, title="Solfege")
    return pitches_cmajor, solfege


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""If the lengths are not factors to each other, the score will expand with interesting patterns.""")
    return


@app.cell
def __(dj, pitches_cmajor):
    _durations = [2, 1, 1] * 8
    _notes = dj.rhythm.isorhythm(pitches=pitches_cmajor, durations=_durations)
    dj.score.show(_notes, title="Isorhythm")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Composition

        I use to name my objects as `track1a_` for the first (`a`) part of track `1`, then the description of the object after the underscore `_`. This helps me to organize my composition. Because the length of the note list (6 items) differs from that of durations (10 items), and durations sum to 8 with the default time signature, each measure will contain the same durations, but with different notes, providing interesting evolving patterns. In the plan, I aimed to create a melody in E major.
        """
    )
    return


@app.cell
def __(dj):
    track1a_p = [68, 64, 71, 69, 75, 73]  # _p for pitch, 30 items in E major
    track1a_d = [
        1,
        0.5,
        0.25,
        0.5,
        1,
        0.75,
        0.5,
        0.5,
        1,
        2,
    ]  # _d for durations, 30 items
    track1a_n = dj.rhythm.isorhythm(pitches=track1a_p, durations=track1a_d)
    dj.score.show(track1a_n)
    return track1a_d, track1a_n, track1a_p


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Additive and subtractive processes""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""A forward additive process on [A, B, C, D] will take the first note, then the first and second, then first, second and third and so on as [A, A, B, A, B, C, A, B, C, D]. Two repetitions will expand the melody more slowly, as [A, A, A, B, A, B, A, B, C, A, B, C, A, B, C, D, A, B, C, D]. Instead of adding the notes gradually, a subtractive process removes them. A forward subtractive process with one repetition will go as [A, B, C, D, B, C, D, C, D, D].""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Example

        To show what is an additive process, let's take the solfège. I will apply additive and subtractive processes, forward and backward on C4, D4, E4, F4, G4, A4, B4, C5. First, the **additive forward process** grows by iteratively adding the next note from the beginning.

        ```
            C4,
            C4, D4,
            C4, D4, E4,
            C4, D4, E4, F4,
            C4, D4, E4, F4, G4,
            C4, D4, E4, F4, G4, A4, 
            C4, D4, E4, F4, G4, A4, B4, 
            C4, D4, E4, F4, G4, A4, B4, C5
        ```
        """
    )
    return


@app.cell
def __(dj, solfege):
    af_process = dj.minimalism.Minimalism.Process(
        operation="additive", direction="forward"
    ).generate(solfege)
    dj.player.show(af_process, tempo=180)
    return (af_process,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        The **additive backward process** grows from the end of the melody and iteratively adds the previous one.

        ```
            C5,
            B4, C5,
            A4, B4, C5,
            G4, A4, B4, C5,
            F4, G4, A4, B4, C5,
            E4, F4, G4, A4, B4, C5,
            D4, E4, F4, G4, A4, B4, C5,
            C4, D4, E4, F4, G4, A4, B4, C5
        ```
        """
    )
    return


@app.cell
def __(dj, solfege):
    ab_process = dj.minimalism.Minimalism.Process(
        operation="additive", direction="backward"
    ).generate(solfege)
    dj.player.show(ab_process, tempo=180)
    return (ab_process,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Subtractive processes inverse the triangles I presented. The **subtractive forward process** plays the whole melody, then iteratively removes the **first** note.

        ```
            C4, D4, E4, F4, G4, A4, B4, C5,
            D4, E4, F4, G4, A4, B4, C5,
            E4, F4, G4, A4, B4, C5,
            F4, G4, A4, B4, C5,
            G4, A4, B4, C5,
            A4, B4, C5,
            B4, C5,
            C5
        ```
        """
    )
    return


@app.cell
def __(dj, solfege):
    sf_process = dj.minimalism.Minimalism.Process(
        operation="subtractive", direction="forward"
    ).generate(solfege)
    dj.player.show(sf_process, tempo=180)
    return (sf_process,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        The **subtractive backward process** plays the whole melody, then iteratively removes the **last** note.

        ```
            C4, D4, E4, F4, G4, A4, B4, C5,
            C4, D4, E4, F4, G4, A4, B4,
            C4, D4, E4, F4, G4, A4,
            C4, D4, E4, F4, G4,
            C4, D4, E4, F4,
            C4, D4, E4,
            C4, D4,
            C4
        ```
        """
    )
    return


@app.cell
def __(dj, solfege):
    sb_process = dj.minimalism.Minimalism.Process(
        operation="subtractive", direction="forward"
    ).generate(solfege)
    dj.player.show(sb_process, tempo=180)
    return (sb_process,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        There are four other use cases involving inward and outward directions in additive and subtracting processes, as well as an option for repetitions, which are covered in the API of the [minimalism module of Djalgo](api.html#djalgo.minimalism.Minimalism). While the outcome of additive and subtractive processes is predictable for simple melodies, complex melodies can expand or shrink to interesting patterns. Unless you have a precise mathematical framework in mind, my suggestion is to empirically try combinations of processes and arguments that sound good to your ears, then investigate what you are really doing with a solfège. However, if the initial melody is long, these processes can expand to very long pieces and become quite boring.

        Let's try additive processes on our composition.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Composition

        I planned to append an additive and a subtractive process to track 1, in E minor. Let's first create a melody.
        """
    )
    return


@app.cell
def __(dj):
    dj.harmony.Scale(tonic="E", mode="major").generate()
    return


@app.cell
def __(dj):
    base_pitches = [71, 64, 80, 73, 85, 83, 78, 63]  # 8 notes in E minor
    base_durations = [1.5, 0.25, 2.25, 1, 0.25, 0.25, 0.5, 2]
    track1b_base = dj.rhythm.isorhythm(
        pitches=base_pitches, durations=base_durations
    )
    dj.score.show(track1b_base)
    return base_durations, base_pitches, track1b_base


@app.cell
def __(mo):
    mo.md(r"""I use an additive forward process and a subtractive backward process. But I prefer to keep durations steady instead of hardly attaching durations to notes.""")
    return


@app.cell
def __(base_durations, dj, itertools, track1b_base):
    track1b_n = dj.minimalism.Minimalism.Process(
        operation="additive", direction="forward"
    ).generate(track1b_base)
    track1b_n.extend(
        dj.minimalism.Minimalism.Process(
            operation="subtractive", direction="backward"
        ).generate(track1b_base)
    )

    track1b_nir = []
    current_offset = 0
    base_duration_cycle = itertools.cycle(base_durations)
    for n in track1b_n:
        duration_n = next(base_duration_cycle)
        track1b_nir.append((n[0], duration_n, current_offset))
        current_offset += duration_n

    dj.score.show(track1b_nir)
    return (
        base_duration_cycle,
        current_offset,
        duration_n,
        n,
        track1b_n,
        track1b_nir,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The minor scale sounds sad and intriguing, and somewhat dissonant. But it sounds incomplete, like a distress without conclusion. I need to conclude the piece like it began, but somewhat differently.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Shuffling

        Shuffling is moving around items in a list. It can be pitches, durations, lists of notes, etc. While shuffling durations will largely change the shape of the melody, shuffling pitches is a reboot of the same rhythmic pattern. Shuffling is the only stochastic (random) process I am going to use in this section. I will use the *random* package, which comes with standard Python installations.

        ### Example

        I create a function that extracts the pitches, shuffles them, then assembles them again in a new list. Just like the `.append()` method, `random.shuffle()` is done *in place* (in place means that the object is modified by the applied method - assigning in-place methods to a new variable will likely cause an error). I use a seed number so that each time I run the random process, I obtain the exact same result.
        """
    )
    return


@app.cell
def __(dj, random, solfege):
    random.seed(4698801)


    def shuffle_pitches(sequence):
        pitches = [note[0] for note in sequence]
        random.shuffle(pitches)
        shuffled_sequence = [
            (pitches[i], note[1], note[2]) for i, note in enumerate(sequence)
        ]
        return shuffled_sequence


    solfege_shuffle = shuffle_pitches(solfege)
    dj.score.show(solfege_shuffle)
    return shuffle_pitches, solfege_shuffle


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Composition

        I use shuffling here to give a sense of completion after the additive / subtractive processes, by randomly moving the notes a `track1a_m`. I changed the seed until I obtain something I like.
        """
    )
    return


@app.cell
def __(dj, random, shuffle_pitches, track1a_n):
    random.seed(38745638)
    track1c_n = shuffle_pitches(track1a_n)
    dj.score.show(track1c_n)
    return (track1c_n,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""It still sounds monotone. A tintinnabuli might enrich the piece.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Tintinnabuli

        The tintinnabuli is a procedural music composition technique developed by the classical composer [Arvo Pärt](https://en.wikipedia.org/wiki/Arvo_P%C3%A4rt) to create spiritual Christian sounds, but amenable to styles at the limit of your imagination. A tintinnabuli is made of two tracks : a melody we call the *m-voice*, and a counterpoint we call the *t-voice* (*t* for tintinnabuli), generated from the *m-voice*.

        The *t-voice* copy-pastes each note from the *m-voice*, then alters the pitch, following a systematic rule used through a musical piece. Rules can really be anything, but the tintinnabuli consists in providing a *t-chord*, and for each note of the *m-voice*, get the pitch in the *t-chord* that is the closest to the pitch of the *m-voice* through a given direction: up, down, any or alternate. The *t-chord* can be anything, but it typically is a major or a minor triad. When you generate your `Tintinnabuli` Python object, for each note in the *m-voice*, the *t-voice* will rank the next (higher pitch) or previous (lower pitch) notes in the *t-chord* depending on the direction you selected, then will select the rank from the position you specified. Each note in the *m-voice* will have its corresponding note in the *t-voice*, with the same duration.

        ### Example

        For instance, I define the t-chord of my t-voice as a C major triad C, E G, I set direction to "up" and the rank to 1. The first note of my melody (m-voice) is a C4. Can you guess the t-voice? You go up from the C4, the next in your chord is E. But you chose position 2, so you look for the next, G. And it will be a G4 since you went up from the C4.

        Let's try the same t-voice properties on a simple monotonic C scale as the m-voice.
        """
    )
    return


@app.cell
def __(dj, solfege):
    m_voice = solfege
    t_voice = dj.minimalism.Minimalism.Tintinnabuli(
        t_chord=[64, 68, 71], direction="up", rank=1
    ).generate(solfege)
    dj.score.show([m_voice, t_voice])
    return m_voice, t_voice


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""While the m-voice grows continuously, the t-voice grows more stepwise. See how the t-voice is restricted to the notes we set in `t_chord = [64, 68, 71]`. You can choose any chord and position and going up with `direction = "up"` or down with `direction = "down"`. Djalgo even allows alternating between up and down with `"alternate"` or to the nearest on `"any"` direction.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Composition

        I will add t-voices to track 1 as the m-voice. I'll add an up t-voice on a E major triad on position 1 for `track1a_m`. For `track1b_m` the t-voice will be down at rank 1 on E minor triad and augmented 7th. And for the *c* part, let's be fancy and use an up alternate on E major augmented 7th with rank 2 while increasing the pitch to a full octave (perfect 8).
        """
    )
    return


@app.cell
def __(dj, track1a_n, track1b_n, track1c_n):
    track2a_n = dj.minimalism.Minimalism.Tintinnabuli(
        t_chord=[64, 68, 71], direction="up"
    ).generate(track1a_n)

    track2b_n = dj.minimalism.Minimalism.Tintinnabuli(
        t_chord=[64, 67, 71, 74], direction="down", rank=1
    ).generate(track1b_n)

    track2c_n = dj.minimalism.Minimalism.Tintinnabuli(
        t_chord=[64, 68, 71, 74], direction="alternate"
    ).generate(track1c_n)
    track2c_n = [(p + 12, d, p) for p, d, o in track2c_n]
    return track2a_n, track2b_n, track2c_n


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Before assembling the a, b and c parts of the tracks, I need to correct the offset since `track1a_n` ends at 24 `track1b_n` starts at zero.""")
    return


@app.cell
def __(track1a_n, track1b_n, track1c_n):
    print("Track 1a", track1a_n[-5:])
    print("Track 1b", track1b_n[:5])

    print("Track 1b", track1b_n[-5:])
    print("Track 1c", track1c_n[:5])
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""In fact, I want `track1b_n` to start at 48 and `track1c_n` to start at 48+36. I will fix this in the next cell.""")
    return


@app.cell
def __(track1b_n, track1c_n, track2b_n, track2c_n):
    track1b_no = [(_p, _d, _o + 24) for _p, _d, _o in track1b_n]
    track1c_no = [(_p, _d, _o + 24 + 36) for _p, _d, _o in track1c_n]

    track2b_no = [(_p, _d, _o + 24) for _p, _d, _o in track2b_n]
    track2c_no = [(_p, _d, _o + 24 + 36) for _p, _d, _o in track2c_n]
    return track1b_no, track1c_no, track2b_no, track2c_no


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""The following block creates the tracks, then assembles them together.""")
    return


@app.cell
def __(
    dj,
    track1a_n,
    track1b_n,
    track1c_n,
    track2a_n,
    track2b_n,
    track2c_n,
):
    track1 = track1a_n + track1b_n + track1c_n
    dj.player.show([track1, track2a_n + track2b_n + track2c_n])
    return (track1,)


@app.cell
def __(mo):
    mo.md(r"""The music still sounds quite dull because the t-voice is always played at the same time steps as the m-voice. Let's be a bit innovative here and offset the t-voice by a quarter length, on part *b* only.""")
    return


@app.cell
def __(dj, track1, track2a_n, track2b_n, track2c_n):
    track2b_offset = [(note[0], note[1], note[2] + 1) for note in track2b_n]
    track2 = track2a_n + track2b_offset + track2c_n
    dj.player.show([track1, track2])
    return track2, track2b_offset


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""I like that! But... it lacks rhythm.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Voicing

        Chords can be used to add a rhythmic structure, a feel of predictability to make my piece sounds less experimental and more enjoyable. A way of procedurally adding chords is to extract the first note of each measure as root for chords. I could extract all first beats per measure, but in cases where the note from the previous measures continues through the next, there is no beat 1. So my strategy is to iteratively split the track at each four quarter lengths, then extract the first note of the second part of the split.

        ### Example

        For the example, I will alter the durations of the solfège. I'll make a copy of it to avoid altering my initial object.
        """
    )
    return


@app.cell
def __(dj, solfege):
    solfege_alterdur = dj.rhythm.isorhythm(
        pitches=[_pitch for _pitch, _, _ in solfege],
        durations=[1, 1.5, 1, 1.5, 1, 1, 0.5, 0.5],
    )
    dj.score.show(solfege_alterdur)
    return (solfege_alterdur,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""A utility function cuts the measures, grabs the first note, then spit out the note with the length of the measure. To dynamically create the adequate number of notes at the beginning of each measure, I computed the number of measures of the `solfege_alterdur` track, given the length of each measure.""")
    return


@app.cell
def __(dj, solfege_alterdur):
    import math  # to compute the rounded to up, i.e. ceiling, math.ceil

    _measure_length = 4
    _end_of_track = (
        solfege_alterdur[-1][2] + solfege_alterdur[-1][1]
    )  # [2] is the offset, [1] is the duration
    _number_of_measures = math.ceil(_end_of_track / _measure_length)
    _closest_pitches = dj.utils.find_closest_pitch_at_measure_start(
        solfege_alterdur, measure_length=_measure_length
    )
    solfege_alterdur_d = [_measure_length] * _number_of_measures
    solfege_alterdur_cp = dj.rhythm.isorhythm(
        pitches=_closest_pitches, durations=solfege_alterdur_d
    )
    dj.score.show(solfege_alterdur_cp)
    return math, solfege_alterdur_cp, solfege_alterdur_d


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Voicing was introduced in the [harmonies](02_harmony.html) section.""")
    return


@app.cell
def __(dj, solfege_alterdur, solfege_alterdur_cp, solfege_alterdur_d):
    _voice = dj.harmony.Voice(tonic="C", mode="major", degrees=[0, 2, 4]).generate(
        notes=[_pitch for _pitch, _, _ in solfege_alterdur_cp]
    )
    solfege_alterdur_voice = dj.rhythm.isorhythm(
        pitches=_voice, durations=solfege_alterdur_d
    )
    dj.score.show([solfege_alterdur, solfege_alterdur_voice])
    return (solfege_alterdur_voice,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Composition

        I created chords for each first notes of the whole track 1, then lower the pitch to an octave (perfect 8). Because I want major chords on *a*, minor chords on *b* and back to major on *c*, I will split the operations in three parts. Note that this is not how music theory really works, and some of the chords will sound dissonant. But I'm fine with some dissonance for this piece, so let this question simmer. Because in my previous operations the tracks are in fact streams of streams, I used the `.flatten()` method, which remove a hierarchical level in streams.
        """
    )
    return


@app.cell
def __(dj, math, track1a_no, track1b_no, track1c_no):
    _measure_length = 4
    track3 = []

    for _track, _mode in zip(
        [track1a_no, track1b_no, track1c_no], ["major", "minor", "major"]
    ):
        _end_of_track = _track[-1][2] + _track[-1][1]
        _number_of_measures = math.ceil(_end_of_track / _measure_length)
        _closest_pitches = dj.utils.find_closest_pitch_at_measure_start(
            _track, measure_length=_measure_length
        )
        _voice_pitches = dj.harmony.Voice(
            tonic="E", mode=_mode, degrees=[0, 2, 4]
        ).generate(pitches=_closest_pitches)
        _voice_notes = dj.rhythm.isorhythm(
            pitches=_voice_pitches,
            durations=[_measure_length] * _number_of_measures,
        )
        track3 = track3 + _voice_notes
    return (track3,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Putting the three tracks together...""")
    return


@app.cell
def __(dj, track1, track2, track3):
    _max_offset = 30
    _filtered_track1 = [_note for _note in track1 if _note[2] < _max_offset]
    _filtered_track2 = [_note for _note in track2 if _note[2] < _max_offset]
    _filtered_track3 = [_note for _note in track3 if _note[2] < _max_offset]
    tracks = []
    tracks.append(_filtered_track1)
    tracks.append(_filtered_track2)
    tracks.append(_filtered_track3)
    dj.player.show(tracks)
    return (tracks,)


@app.cell
def __(mo):
    mo.md(r"""It's kind of messy, but sounds pretty good to me! Since I'm satisfied with this piece, I export the tracks to the midi format in the aim of import them in my digital audio workstation (DAW).""")
    return


@app.cell
def __(dj, tracks):
    # !mkdir music
    dj.player.show(tracks)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""I used free instruments of Spitfire Audio LABS and Tracktion Waveform Free as DAW to create the piece from our midi files. These are free, so if you want to play with midi files and rich sounds, these pieces of software are very good starts. After some assembly, duplicated tracks to different instruments and mastering, I published my piece on Soundcloud.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""<iframe width="100%" height="150" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1295425996&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/user-512016957-418252282" title="motife" target="_blank" style="color: #cccccc; text-decoration: none;">motife</a> · <a href="https://soundcloud.com/user-512016957-418252282/arvo-ensemble-006" title="Arvo ensemble 006" target="_blank" style="color: #cccccc; text-decoration: none;">Arvo ensemble 006</a></div>""")
    return


@app.cell
def __(mo):
    mo.md(r"""↳ [Walks](05_walks.html)""")
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
