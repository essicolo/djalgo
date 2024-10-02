import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import djalgo as dj
    import marimo as mo

    instrument1 = [
        (60, 0.5, 0),
        (62, 0.5, 0.5),
        (64, 0.5, 1),
        (65, 0.5, 1.5),
        (67, 0.5, 2),
        (69, 0.5, 2.5),
        (71, 0.5, 3),
        (72, 0.5, 3.5),
    ]

    instrument2 = [
        (72, 0.5, 0),
        (71, 0.5, 0.5),
        (69, 0.5, 1),
        (67, 0.5, 1.5),
        (65, 0.5, 2),
        (64, 0.5, 2.5),
        (62, 0.5, 3),
        (60, 0.5, 3.5),
    ]


    player = dj.player.show(
        [instrument1, instrument2],
        initial_bpm=120,
    )
    player
    return dj, instrument1, instrument2, mo, player


@app.cell
def __(dj, instrument1, instrument2):
    dj.score.show([instrument1, instrument2], title="Wonderlous")
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
