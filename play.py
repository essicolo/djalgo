import marimo

__generated_with = "0.9.4"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import djalgo as dj
    return dj, mo


@app.cell
def __():
    return


@app.cell
def __(dj):
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
        (60, 1.0, 0.0),  # C (twin)
        (60, 1.0, 1.0),  # C (kle)
        (67, 1.0, 2.0),  # G (twin)
        (67, 1.0, 3.0),  # G (kle)
        (69, 1.0, 4.0),  # A (lit)
        (69, 1.0, 5.0),  # A (tle)
        (67, 2.0, 6.0),  # G (star)
    ]
    dj.player.show([twinkle_1, twinkle_2])
    return twinkle_1, twinkle_2


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
