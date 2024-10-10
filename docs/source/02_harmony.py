import marimo

__generated_with = "0.9.4"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# 2. Harmonies""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2.1 Python basics

        ### 2.1.1 Lists in First Class

        Djalgo uses basic Python objects such as lists and tuples. Python lists won't produce music, but they rock after parsing them through MIDI processors or synthesizers. The content of a Python list is defined in square brackets, and each item is separated with a comma. In the next code cell, I assign a list to a variable.
        """
    )
    return


@app.cell
def __():
    a = [1, "a", 10, "crocodile"]
    a
    return (a,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### 2.1.2 Music as a signal of information

        Rather than sounds, Djalgo generates numerical values representing notes. A note, at its most essential information, is a combination of a pitch, a duration, and when it starts in time. Djalgo considers a note as a (pitch, duration, offset) tuple. Pitches are expressed in MIDI notation, a highly normed and complex way for music encoding, which spans from 0, corresponding to C2 (8.178 Hz), to 127, corresponding to G9 (12543.854 Hz). Durations, as well as offsets, or start times, are expressed in quarter lengths. A quarter length is the duration of a metronome tick. The metronome tick oscillates in beats per minute, a speed that allows quarter lengths to be placed in time.

        In Python, tuples are immutable lists: once a tuple is defined, it can't be altered. The tuple `(72, 2.0, 1.0)` defines a note with pitch C4 with a duration of two quarter lengths starting at 1.0 quarter length from the beginning of the track. Pitches defined by `None` are rests. 

        A rhythm is simply a note without the pitch. It is defined with a tuple of `(duration, offset)`. Speaking of definitions, a track is a sequence of notes stored in a list. And multiple tracks form a piece, which becomes a list of lists.

        Let's define two tracks.
        """
    )
    return


@app.cell
def __():
    twinkle_1 = [
        (60, 1.0, 0.0),  # C (twin)
        (60, 1.0, 1.0),  # C (kle)
        (67, 1.0, 2.0),  # G (twin)
        (67, 1.0, 3.0),  # G (kle)
        (69, 1.0, 4.0),  # A (lit)
        (69, 1.0, 5.0),  # A (tle)
        (67, 2.0, 6.0),  # G (star)
    ]

    twinkle_2 = [
        (65, 1.0, 8.0),  # F (how)
        (65, 1.0, 9.0),  # F (I)
        (64, 1.0, 10.0),  # E (won)
        (64, 1.0, 11.0),  # E (der)
        (62, 1.0, 12.0),  # D (what)
        (62, 1.0, 13.0),  # D (you)
        (60, 2.0, 14.0),  # C (are)
    ]
    return twinkle_1, twinkle_2


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""To merge two lists *horizontally*, i.e. in the time direction, you can use the `+` opetator.""")
    return


@app.cell
def __(twinkle_1, twinkle_2):
    twinkle_1 + twinkle_2
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Stack them *vertically* creates a piece of two tracks.""")
    return


@app.cell
def __(twinkle_1, twinkle_2):
    [twinkle_1, twinkle_2]
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2.2 Leverage Djalgo for music composition

        ### 2.2.1 Scales

        We haven't used Djalgo yet. We just played with basic Python where I wrote the song *Twinkle, Twinkle Little Star* in C-major. C-major is a scale, i.e. a subset of the chromatic scale (all pitches) designed to fit together. Djalgo can generate pitch lists allowed for a given scale. We'll need to load Djalgo in our session to access to its functionalities. I use the alias `dj` to make the code shorter.
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
        Scales are accessible from the *harmony* module. You have to define the tonic and the type, then `.generate()` will process the scale, returning all available MIDI pitches in the scale. The `object.method()` way of programming (object-oriented programming) is just like defining a frog and make it jump, as

        ```
        frog = animal(order='anura')
        frog.jump()
        frog.swim()
        ```

        In the following code block, I seek for the scale function in Djalgo, define it, then tell it to generate the scale.
        """
    )
    return


@app.cell
def __(dj):
    c_major = dj.harmony.Scale(tonic="C", mode="major").generate()
    c_major
    return (c_major,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Scales are defined as intervals from the chromatic scale. You might have heard that a major scale is *whole-step, whole-step, half-step, whole-step, whole-step, whole-step, half-step*. In Python, from a list of 12 pitches in the chromatic scale, you would take the first pitch (index 0), the third (index 2), and so on. Djalgo predefines the major scale, the minor, diminished, pentatonic and so on.""")
    return


@app.cell
def __(dj):
    dj.harmony.Scale.scale_intervals
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""As any list, you can extract a subset by index. In Python, `c_major[35:43]` means you aim at extracting index 35 to *excluding* index 43, i.e. indexes 35 to 42. The resulting list is C4 to C5.""")
    return


@app.cell
def __(c_major):
    c_major[35:43]  # C4 to C5
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""To convert a list of pitches to the Djalgo notation, we could use afor loop. The explanations are in code comments, which are placed after the `#` sign.""")
    return


@app.cell
def __(c_major):
    # Initialize an empty list to store the notes
    c_major_sub_notes = []

    # Initialize the offset, the first being 0
    _durations = [1] * len(c_major[35:43])
    _offsets = [sum(_durations[: i + 1]) for i in range(len(_durations))]

    # Iterate over the pitches in the scale subset we assigned earlier
    for _p, _d, _o in zip(c_major[35:43], _durations, _offsets):
        c_major_sub_notes.append((_p, _d, _o))

    c_major_sub_notes
    return (c_major_sub_notes,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We can use `dj.score.show()` function to render the score.""")
    return


@app.cell
def __(c_major_sub_notes, dj):
    # dj.conversion.convert(c_major_sub_notes, to='music21').show() # example for music21 and MuseScore
    dj.score.show(c_major_sub_notes, title="C Major Scale", key="C-major")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Another way to show music is by using the built-in music player.""")
    return


@app.cell
def __(c_major_sub_notes, dj):
    dj.player.show(c_major_sub_notes)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Chords

        A chord is multiple pitches played together, generally three. In Djalgo, chords are written as a list of pitches in the note format.
        """
    )
    return


@app.cell
def __(dj):
    c_major_chord = ([60, 64, 67], 1, 0)
    dj.score.show([c_major_chord], title="C-major chord")
    return (c_major_chord,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Ornaments

        We used the Djalgo package, but still haven't seen how it can help to generate music. Let's start with ornaments, which alter a list of notes to create a richer score. Djalgo has six types of ornaments: grace note, trill, mordent, arpeggio, turn and slide.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""**Grace note** adds a note randomly drawned from the list given in `grace_pitches` at the place given by `note_index`.""")
    return


@app.cell
def __(c_major_sub_notes, dj):
    _ornam = dj.harmony.Ornament(
        type="grace_note", grace_note_type="appoggiatura", grace_pitches=[72]
    ).generate(notes=c_major_sub_notes, note_index=4)
    dj.score.show(_ornam, title="Grace note")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""**Trill** gets the degree given by `by` from the note at `note_index` and oscillates at rate of `trill_rate` between the note and its degree.""")
    return


@app.cell
def __(c_major_sub_notes, dj):
    _ornam = dj.harmony.Ornament(
        type="trill", trill_rate=0.125, by=1, tonic="C", mode="major"
    ).generate(notes=c_major_sub_notes, note_index=4)
    dj.score.show(_ornam, title="Trill")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""**Mordent** rapidly alternates between the original pitch and one step defined `by`.""")
    return


@app.cell
def __(c_major_sub_notes, dj):
    _ornam = dj.harmony.Ornament(
        type="mordent", by=-1, tonic="C", mode="major"
    ).generate(notes=c_major_sub_notes, note_index=4)
    dj.score.show(_ornam, title="Mordent")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""**Arpeggio** transforms a note to an arpeggio given by a list of degrees.""")
    return


@app.cell
def __(c_major_sub_notes, dj):
    _ornam = dj.harmony.Ornament(
        type="arpeggio", tonic="C", mode="major", arpeggio_degrees=[0, 4, 2, 5]
    ).generate(notes=c_major_sub_notes, note_index=4)
    dj.score.show(_ornam, title="Arpeggio")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""**Turn** is a transition of four notes between `note_index` and the next note.""")
    return


@app.cell
def __(c_major_sub_notes, dj):
    _ornam = dj.harmony.Ornament(type="turn", tonic="C", mode="major").generate(
        notes=c_major_sub_notes, note_index=4
    )
    dj.score.show(_ornam, title="Turn")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""**Silde** is a glissando. However, in Djalgo, glissandos should be defined at the instrument level with your preferred package (`Instrument` in Pretty-midi and `Stream` in Music21). Instead of sliding, slide in djalgo transits on the chromatic scale from a note to the next.""")
    return


@app.cell
def __(dj):
    _ornam = dj.harmony.Ornament(type="slide", slide_length=6).generate(
        notes=[(60, 4, 0), (72, 4, 4)], note_index=0
    )
    dj.score.show(_ornam, title="Slide")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Voice

        Voicing creates chords from pitch lists. These are just lists, but iterating through them can generate either chords and arpeggios.
        """
    )
    return


@app.cell
def __(c_major_sub_notes, dj):
    pitch_chords = dj.harmony.Voice(
        tonic="C",
        mode="major",
        degrees=[0, 2, 4],  # triads
    ).generate(notes=c_major_sub_notes)
    dj.score.show([c_major_sub_notes, pitch_chords], title="Voicing")
    # print(pitch_chords)
    return (pitch_chords,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Progression

        Ever heard of the circle of fifths? It can be used to create progressions with chords that fit together. Why not using it to generate random progressions from different circles, the circle of fifths (`'P5'`) being the most popular. The radius argument is the spread of the chords in the circle across [major chords, minor chords, diminished chords], usually `[3, 3, 1]`.
        """
    )
    return


@app.cell
def __(dj):
    progression1 = dj.harmony.Progression(
        tonic_pitch="D3", circle_of="P5", radius=[3, 3, 1]
    ).generate(
        length=8, seed=5
    )  # a seed is any random integer that allows you to reproduce the same outcomes from arandom process
    progression1_notes = []
    _offset = 0
    for _chord in progression1:
        progression1_notes.append((_chord, 1, _offset))
        _offset = _offset + 1
    dj.score.show(progression1_notes, title="Random progression 1")
    return progression1, progression1_notes


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Rhythms

        For now, all note durations was set to 1 quarter length, and offsets were set accordingly. A combination of durations and offsets is called a rhythm in Djalgo. Rhythms can be set by hand, but to leverage Djalgo, we can generate them randomly. The `random` method from the `rhythm` module draws numbers from a `durations` list until they sum up to the `measure_length`.
        """
    )
    return


@app.cell
def __(dj):
    random_rhythm = dj.rhythm.Rhythm(
        measure_length=8, durations=[0.5, 1, 2]
    ).random(seed=3)
    random_rhythm
    return (random_rhythm,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""A random progression of the same length can be generated (`len(a)` take the length of the list `a`), mapped to the rhythm, and transformed to a Music21 stream to create a score or a midi.""")
    return


@app.cell
def __(dj, random_rhythm):
    progression2 = dj.harmony.Progression(
        tonic_pitch="C3", circle_of="P5", radius=[3, 3, 1]
    ).generate(length=len(random_rhythm), seed=5)
    random_progression2 = [
        (p, d, o) for p, (d, o) in zip(progression2, random_rhythm)
    ]
    dj.score.show(random_progression2, title="Random progression 2")
    return progression2, random_progression2


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Wrap up

        Let's take our twinkle song.
        """
    )
    return


@app.cell
def __(dj, twinkle_1, twinkle_2):
    twinkle = twinkle_1 + twinkle_2
    dj.score.show(twinkle, title="Twinkle Twinkle Little Star")
    return (twinkle,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""We will add a voice to index 0 and index 7.""")
    return


@app.cell
def __(dj, twinkle):
    twinkle_chords = dj.harmony.Voice(
        tonic="C",
        mode="major",
        degrees=[0, 2, 4],  # triads
    ).generate(notes=twinkle)
    twinkle_chords = [twinkle_chords[0], twinkle_chords[7]]
    dj.score.show(
        [twinkle, twinkle_chords], title="Twinkle Twinkle Little Star with chords"
    )
    return (twinkle_chords,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Some ornaments...""")
    return


@app.cell
def __(dj, twinkle, twinkle_chords):
    twinkle_o = dj.harmony.Ornament(
        type="trill", trill_rate=0.25, by=1, tonic="C", mode="major"
    ).generate(notes=twinkle, note_index=6)

    twinkle_o = dj.harmony.Ornament(
        type="mordent", trill_rate=0.25, by=1, tonic="C", mode="major"
    ).generate(notes=twinkle_o, note_index=len(twinkle) - 1)

    dj.score.show([twinkle_o, twinkle_chords], title="Twinkle, with ornaments")
    return (twinkle_o,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""To avoid an abrupt ending, let's alter the last duration to 4.667, so that the last note of the mordent ends up its measure and lasts another measure.""")
    return


@app.cell
def __(twinkle):
    twinkle[-1] = (twinkle[-1][0], 4.667, twinkle[-1][2])
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""And let's hear our masterpiece!""")
    return


@app.cell
def __(dj, twinkle, twinkle_chords):
    dj.player.show([twinkle, twinkle_chords])
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        If you run Python locally, another way to hear it is using the (wonderful) SCAMP library. Make sure It's installed with `pip install djalgo[musician]`. 

        ```
        import scamp # install it first!
        s = scamp.Session(tempo=120)
        instrument = s.new_part('flute')
        for i,n in enumerate(twinkle):
            # play either a note or a chord
            if isinstance(n[0], list):
                instrument.play_chord(n[0], 1, n[1])
            else:
                instrument.play_note(n[0], 1, n[1])
            # the following is not necessary here, it just assures that rests are respected
            #if i < (len(twinkle)-1):
            #    scamp.wait(twinkle[i+1][2]-(n[1] + n[2]))
        s.wait_for_children_to_finish() # end the SCAMP session in notebooks
        ```
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        You should now be able to modify a piece, decorate it, and export it. Next step:

        ↳ [Loops](03_loops.html)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
