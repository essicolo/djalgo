import anywidget
import traitlets
from typing import List, Dict, Union, Tuple

class MusicPlayer(anywidget.AnyWidget):
    tracks = traitlets.List().tag(sync=True)
    custom_instruments = traitlets.Dict().tag(sync=True)
    initial_bpm = traitlets.Int(default_value=120).tag(sync=True)
    show_debug = traitlets.Bool(default_value=False).tag(sync=True)
    _esm = """
export async function render({ model, el }) {
    const colors = {
        background: '#FDF6E3',
        primary: '#333',
        secondary: '#EEE8D5',
        accent: '#333',
        text: '#073642',
        lightText: '#586E75'
    };

    const styles = {
        container: `
            font-family: 'Georgia', serif;
            background-color: ${colors.background};
            color: ${colors.text};
            padding: 20px;
            border-radius: 12px;
            width: 400px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `,
        controlsContainer: `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        `,
        select: `
            padding: 8px;
            margin: 5px;
            border: none;
            border-radius: 6px;
            background-color: ${colors.secondary};
            color: ${colors.text};
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        `,
        button: `
            flex: 0 0 80px;
            padding: 10px;
            border: none;
            border-radius: 6px;
            background-color: ${colors.primary};
            color: ${colors.background};
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        `,
        bpmInput: `
            width: 60px;
            padding: 8px;
            margin: 0 10px;
            border: none;
            border-radius: 6px;
            background-color: ${colors.secondary};
            color: ${colors.text};
            font-size: 16px;
            text-align: center;
        `,
        timelineContainer: `
            position: relative;
            width: 100%;
            margin: 20px 0;
        `,
        timeline: `
            width: 100%;
            -webkit-appearance: none;
            background: ${colors.secondary};
            outline: none;
            border-radius: 15px;
            overflow: visible;
            height: 8px;
        `,
        timeDisplay: `
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: ${colors.text};
            margin-top: 5px;
        `,
        buttonContainer: `
            display: flex;
            justify-content: space-between;
            margin-top: 15px;
        `,
        downloadButton: `
            padding: 10px 20px;
            margin: 0 5px;
            border: none;
            border-radius: 6px;
            background-color: ${colors.accent};
            color: ${colors.background};
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            flex: 1;
        `
    };

    const createElementWithStyle = (tag, style) => {
        const element = document.createElement(tag);
        element.style.cssText = style;
        return element;
    };

    const container = createElementWithStyle('div', styles.container);
    const controlsContainer = createElementWithStyle('div', styles.controlsContainer);
    const synthSelectors = document.createElement('div');
    const bpmInput = createElementWithStyle('input', styles.bpmInput);
    Object.assign(bpmInput, { type: 'number', min: 60, max: 240, value: model.get('initial_bpm') });
    const playButton = createElementWithStyle('button', styles.button);
    playButton.textContent = '▶';
    const timelineContainer = createElementWithStyle('div', styles.timelineContainer);
    const timelineSlider = createElementWithStyle('input', styles.timeline);
    Object.assign(timelineSlider, { type: 'range', min: 0, max: 100, value: 0 });
    const timeDisplay = createElementWithStyle('div', styles.timeDisplay);
    const currentTime = document.createElement('span');
    currentTime.textContent = '0:00';
    const totalTime = document.createElement('span');
    totalTime.textContent = '0:00';
    timeDisplay.append(currentTime, totalTime);

    const showDebug = model.get('show_debug');
    const debugDiv = document.createElement('div');

    const updateDebug = msg => {
        if (showDebug) {
            model.set('debug_msg', msg);
            model.save_changes();
            debugDiv.textContent = msg;
        }
    };

    // Load Tone.js and @tonejs/midi dynamically
    await Promise.all([
        import('https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js'),
        import('https://unpkg.com/@tonejs/midi@2.0.28/build/Midi.js')
    ]);
    updateDebug('Tone.js and @tonejs/midi loaded');

    const customInstruments = model.get('custom_instruments');
    const synthTypes = ['Synth', 'AMSynth', 'FMSynth', 'MembraneSynth', ...Object.keys(customInstruments)];
    const synthSelectsArray = [];

    let musicData = model.get('tracks');
    musicData = Array.isArray(musicData[0]) ? musicData : [musicData];

    musicData.forEach((_, index) => {
        const synthSelectorItem = document.createElement('div');
        const synthLabel = document.createElement('label');
        synthLabel.textContent = `Instrument ${index + 1}`;
        const synthSelect = createElementWithStyle('select', styles.select);
        synthTypes.forEach(synth => {
            const option = document.createElement('option');
            option.value = synth;
            option.textContent = synth;
            synthSelect.appendChild(option);
        });
        synthSelectsArray.push(synthSelect);
        synthSelectorItem.append(synthLabel, synthSelect);
        synthSelectors.appendChild(synthSelectorItem);
    });

    let synths = [];
    let parts = [];
    let totalDuration;

    const formatTime = seconds => `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;

    const createCustomInstrument = config => new Tone.Synth(config).toDestination();

    const initAudio = () => {
        synths.forEach(s => s.dispose());
        parts.forEach(p => p.dispose());
        synths = [];
        parts = [];

        musicData.forEach((instrumentData, index) => {
            const selectedSynth = synthSelectsArray[index].value;
            const synth = customInstruments[selectedSynth] 
                ? createCustomInstrument(customInstruments[selectedSynth].config)
                : new Tone[selectedSynth]().toDestination();
            synths.push(synth);

            const part = new Tone.Part((time, note) => {
                if (note.noteName === null) {
                    // C'est un silence, ne rien jouer
                    return;
                }
                if (Array.isArray(note.noteName)) {
                    // C'est un accord
                    note.noteName.forEach(n => {
                        if (n !== null) {
                            synth.triggerAttackRelease(n, note.duration, time);
                        }
                    });
                } else {
                    // C'est une note simple
                    synth.triggerAttackRelease(note.noteName, note.duration, time);
                }
            }, instrumentData.map(([midi, duration, time]) => ({
                time,
                noteName: midi === null ? null :
                    (Array.isArray(midi) 
                        ? midi.map(m => m !== null ? Tone.Frequency(m, "midi").toNote() : null)
                        : Tone.Frequency(midi, "midi").toNote()),
                duration
            }))).start(0);
            parts.push(part);
        });

        totalDuration = Math.max(...musicData.flat().map(note => note[2] + note[1]));
        Tone.Transport.loopEnd = totalDuration;
        Tone.Transport.loop = true;

        totalTime.textContent = formatTime(totalDuration);
        updateDebug(`Audio initialized, duration: ${totalDuration.toFixed(2)}s`);
    };

    const updateTimeline = () => {
        const progress = (Tone.Transport.seconds / totalDuration) * 100;
        timelineSlider.value = progress;
        currentTime.textContent = formatTime(Tone.Transport.seconds);
        if (Tone.Transport.state === 'started') {
            requestAnimationFrame(updateTimeline);
        }
    };

    playButton.onclick = async () => {
        try {
            if (Tone.Transport.state === 'started') {
                await Tone.Transport.stop();
                playButton.textContent = '▶';
                updateDebug('Playback stopped');
            } else {
                if (synths.length === 0) initAudio();
                await Tone.start();
                await Tone.Transport.start();
                playButton.textContent = '◼';
                updateDebug('Playback started');
                updateTimeline();
            }
        } catch (error) {
            updateDebug(`Error: ${error.message}`);
        }
    };

    synthSelectsArray.forEach(select => {
        select.onchange = initAudio;
    });

    timelineSlider.oninput = () => {
        const time = (timelineSlider.value / 100) * totalDuration;
        Tone.Transport.seconds = time;
        currentTime.textContent = formatTime(time);
    };

    bpmInput.onchange = () => {
        const bpm = parseInt(bpmInput.value);
        if (bpm >= 60 && bpm <= 240) {
            Tone.Transport.bpm.value = bpm;
            updateDebug(`BPM set to ${bpm}`);
        } else {
            bpmInput.value = Tone.Transport.bpm.value;
            updateDebug('Invalid BPM. Please enter a value between 60 and 240.');
        }
    };

    const buttonContainer = createElementWithStyle('div', styles.buttonContainer);

    const downloadMIDIButton = createElementWithStyle('button', styles.downloadButton);
    downloadMIDIButton.textContent = 'MIDI';

    const generateMIDI = () => {
        const midi = new Midi();
        const bpm = Tone.Transport.bpm.value;
        musicData.forEach(instrumentData => {
            const track = midi.addTrack();
            instrumentData.forEach(([midiNote, duration, time]) => {
                if (midiNote === null) {
                    // C'est un silence, ne rien ajouter à la piste MIDI
                    return;
                }
                if (Array.isArray(midiNote)) {
                    // C'est un accord
                    midiNote.forEach(note => {
                        if (note !== null) {
                            track.addNote({ midi: note, time, duration });
                        }
                    });
                } else {
                    // C'est une note simple
                    track.addNote({ midi: midiNote, time, duration });
                }
            });
        });
        midi.header.setTempo(bpm);
        return midi.toArray();
    };

    downloadMIDIButton.onclick = () => {
        const midiData = generateMIDI();
        const blob = new Blob([midiData], { type: "audio/midi" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'generated_music.mid';
        link.click();
        URL.revokeObjectURL(url);
        updateDebug('MIDI file downloaded');
    };

    const downloadWavButton = createElementWithStyle('button', styles.downloadButton);
    downloadWavButton.textContent = 'WAV';

    const audioBufferToWav = buffer => {
        const numOfChan = buffer.numberOfChannels;
        const length = buffer.length * numOfChan * 2 + 44;
        const out = new ArrayBuffer(length);
        const view = new DataView(out);
        let offset = 0;
        let pos = 0;

        const setUint16 = data => {
            view.setUint16(pos, data, true);
            pos += 2;
        };

        const setUint32 = data => {
            view.setUint32(pos, data, true);
            pos += 4;
        };

        setUint32(0x46464952);
        setUint32(length - 8);
        setUint32(0x45564157);
        setUint32(0x20746d66);
        setUint32(16);
        setUint16(1);
        setUint16(numOfChan);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * numOfChan);
        setUint16(numOfChan * 2);
        setUint16(16);
        setUint32(0x61746164);
        setUint32(length - pos - 4);

        const channels = Array.from({length: buffer.numberOfChannels}, (_, i) => buffer.getChannelData(i));

        while (pos < length) {
            channels.forEach(channel => {
                const sample = Math.max(-1, Math.min(1, channel[offset]));
                view.setInt16(pos, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                pos += 2;
            });
            offset++;
        }

        return out;
    };

    const generateWAV = async () => {
        const duration = Math.max(...musicData.flat().map(note => note[2] + note[1]));

        const buffer = await Tone.Offline(({ transport }) => {
            musicData.forEach((instrumentData, index) => {
                const synthType = synthSelectsArray[index].value;
                const synth = customInstruments[synthType] 
                    ? createCustomInstrument(customInstruments[synthType].config)
                    : new Tone[synthType]().toDestination();

                new Tone.Part((time, note) => {
                    if (note.noteName === null) {
                        // C'est un silence, ne rien jouer
                        return;
                    }
                    if (Array.isArray(note.noteName)) {
                        // C'est un accord
                        note.noteName.forEach(n => {
                            if (n !== null) {
                                synth.triggerAttackRelease(n, note.duration, time);
                            }
                        });
                    } else {
                        // C'est une note simple
                        synth.triggerAttackRelease(note.noteName, note.duration, time);
                    }
                }, instrumentData.map(([midi, duration, time]) => ({
                    time,
                    noteName: midi === null ? null :
                        (Array.isArray(midi) 
                            ? midi.map(m => m !== null ? Tone.Frequency(m, "midi").toNote() : null)
                            : Tone.Frequency(midi, "midi").toNote()),
                    duration
                }))).start(0);
            });

            transport.start();
        }, duration);

        return new Blob([audioBufferToWav(buffer.get())], { type: 'audio/wav' });
    };

    downloadWavButton.onclick = async () => {
        updateDebug('Generating WAV file...');
        const wavBlob = await generateWAV();
        const url = URL.createObjectURL(wavBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'generated_music.wav';
        link.click();
        URL.revokeObjectURL(url);
        updateDebug('WAV file downloaded');
    };

    container.appendChild(synthSelectors);
    controlsContainer.append(bpmInput, playButton);
    container.appendChild(controlsContainer);
    timelineContainer.append(timelineSlider, timeDisplay);
    container.appendChild(timelineContainer);
    buttonContainer.append(downloadMIDIButton, downloadWavButton);
    container.appendChild(buttonContainer);

    if (showDebug) {
        container.appendChild(debugDiv);
    }
    el.appendChild(container);

    // Set initial BPM
    Tone.Transport.bpm.value = model.get('initial_bpm');
    updateDebug('Widget initialized');
}

export default { render };
    """

@staticmethod
def show(
    tracks: Union[List[Tuple[Union[int, List[int], None], float, float]], List[List[Tuple[Union[int, List[int], None], float, float]]]],
    custom_instruments: Dict[str, Dict] = None,
    initial_bpm: int = 120
) -> anywidget.AnyWidget:
    """
    Crée un lecteur de musique avancé.

    Args:
        tracks: Données musicales sous forme de liste de tuples (note MIDI, durée, temps) ou liste de listes de tuples.
        custom_instruments: Dictionnaire d'instruments personnalisés.
        initial_bpm: BPM initial.

    Returns:
        Un widget marimo contenant le lecteur de musique avancé.
    """
    if not isinstance(tracks[0], list):
        tracks = [tracks]

    custom_instruments = custom_instruments or {}

    return (
        MusicPlayer(
            tracks=tracks,
            custom_instruments=custom_instruments,
            initial_bpm=initial_bpm,
            debug_msg="Widget initialized",
        )
    )