import marimo

__generated_with = "0.9.4"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 8. Machine learning

        We introduced machine learning while fitting Gaussian processes in section [5. Walks](05_walks.html). Djalgo's module `djai` includes tools for modelling music from MIDI data relying on PyTorch (a package for deep learning) and MidiTok (a package to transform MIDI files to deep learning-readable format). `djai` is not loaded by default when importing Djalgo, since otherwise PyTorch and MidiTok, which are complicated packages, should have been added to Djalgo's dependencies.

        To use `djai`, you must [`pip install torch`](https://pytorch.org/get-started/locally/) and [`pip install miditok`](https://miditok.readthedocs.io/) in your environment. Because code on marimo.app runs in the browser, and the version of Python running in the browser (Pyodide) can't understand the backend of Pytorch (mostly C++), `torch` won't install on marimo.app. You'll have to run this in a cloud environment (such as [Deepnote](https://deepnote.com/)) or locally.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ethics: art as the witnesses of experience

        Even though `djai` was the module which took me the most time to develop, it is these days, to my opinion, the least interesting. Who needs to DIY their own AI when interesting results can already be generated with a command prompt to a large language model (LLM)? My ethos will fluctuate and evolve, as anything should in the precious, short time we exist. There is nothing inherently wrong with AI, but if your piece was generated with a banal command prompt, your creative process is anything but banal and uninteresting, no matter the result. In times when any artistic piece needed years of work, the result was more important than the process. Now, when anyone can ask a LLM to generate an image of a cat riding a dinosaur in space in the style of a mix of Daly and cyber-punk, well, results are generated within seconds, and the process becomes more relevant. The process can, of course, be interesting *and* imply AI. Indeed, if like me, you have spent months to design your own AI (which is still not working so well...), the *process* (not the result) behind the musical piece has an artistic value as good as any composer who has spent those months studying musical theory. Let's also keep in mind that the process includes both the originality of the approach and the enjoyment of the artist.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Artists are people who spent the precious time they own to think on the narration of the object they created. When the process becomes applying a recepe, the result quits art ant belongs to the same category of home sweet home printed carpets sold on Amazon.

        That's why the `djai` module doesn't come with pre-trained models. That would have been too easy, right? I prefer seeing you tweak it and train it with your own compositions rather than just use it on Leonard Cohen's songs to generate new ones. You worth more than this, and the world deserves more than command-prompt artists.

        > In the quiet moments between the shadow and the light, we find the songs that our hearts forgot to sing. — *"Write an original quote in the style of Leonard Cohen", sent to ChatGPT-4.*
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""<iframe src="https://indiepocalypse.social/@AuthorJMac/112178826967890119/embed" class="mastodon-embed" style="max-width: 100%; border: 0" width="600" height="280" allowfullscreen="allowfullscreen"></iframe><script src="https://indiepocalypse.social/embed.js" async="async"></script>""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Djai

        At the core of Djai, you'll find the `ModelManager`, doing almost everything for you: it scans your midi files, tokenize (prepare for modelling), models them (defines the model), and predicts (generates a midi file). Let's create an instance of the model, then I'll explain the arguments.

        ```
        from djalgo import djai
        model_manager = djai.ModelManager(
            sequence_length_input=24, sequence_length_output=8,
            model_type='gru', nn_units=(64, 64, 64), dropout=0.25,
            batch_size=32, learning_rate=0.001
        )
        ```
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        1. `sequence_length_input`: This defines the length of the input sequences fed into the model. In this case, it is set to 24, meaning each input sequence will consist of 24 tokens.
        2. `sequence_length_output`: This specifies the length of the output sequences generated by the model. Here, it is set to 8, so the model will generate sequences with 8 tokens as output. With `sequence_length_input=24` and `sequence_length_output=8`, each 24 tokens (notes) generates 8 tokens.
        3. `model_type`: This argument indicates the type of neural network model to be used. Possible values include 'gru', 'lstm', and 'transformer'. In this example, 'gru' specifies that a GRU (Gated Recurrent Unit) model will be used. To be short, 

          - LSTMs (Long Short-Term Memory networks) are more traditional and capable but tend to be complex.
          - GRUs (Gated Recurrent Units) aim to simplify the architecture of LSTMs with fewer parameters while maintaining performance.
          - Transformers are at the forefront of current large language model (LLM) technology, offering potentially superior learning capabilities due to their attention mechanisms, albeit at the cost of increased complexity and computational demands.

        4. `nn_units`: This tuple defines the number of units in each layer of the neural network. For the GRU model, (64, 64, 64) means there are three layers, each with 64 units. The more units and layers you'll add, the longer your model will take time to get fitted. Too few units and layers, and your model will not perform well (underfitting). Too many units and layers, and your model will think noise is a trend (overfitting).
        5. `dropout`: This is the dropout rate applied during training to prevent overfitting. A value of 0.25 means that 25% of the units will be randomly dropped during training.
        6. `batch_size`: This determines the number of samples per batch of input fed into the model during training. A batch_size of 32 indicates that 32 sequences will be processed together in each training step.
        7. `learning_rate`: This is the learning rate for the optimizer, which controls how much to adjust the model's weights with respect to the loss gradient. A lower learning rate of 0.001 is used to make finer updates to the weights, potentially leading to better convergence.
        8. `n_heads`: This argument is specific to the transformer model and defines the number of attention heads in each multi-head attention layer. It is not applicable to the GRU model.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        You will have to put the midi files you aim to train your model aim in a folder, here `_midi-djai`.


        ```
        from pathlib import Path
        midi_files = list(Path('_midi-djai').glob('*.mid'))
        ```

        All you'll have to do is to fit our model, save it for eventual future use (large models can take a long time to converge), and generate a new midi file from any midi file used as primer.

        ```
        model_manager.fit('_midi-djai', epochs=500, verbose=25)
        model_manager.save('_midi-djai/gru.model')
        model_manager.generate(length=10, primer_file='_midi-output/polyloop.mid', output_file='_midi-output/djai.mid')
        ```
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()