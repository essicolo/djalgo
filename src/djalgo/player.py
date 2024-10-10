import anywidget
import traitlets
from typing import List, Dict, Union, Tuple

class MusicPlayer(anywidget.AnyWidget):
    tracks = traitlets.List().tag(sync=True)
    custom_instruments = traitlets.Dict().tag(sync=True)
    tempo = traitlets.Int(default_value=120).tag(sync=True)
    show_debug = traitlets.Bool(default_value=False).tag(sync=True)
    debug_msg = traitlets.Unicode().tag(sync=True)

    _esm = """
export async function render({ model, el }) {
    const colors = {
        background: '#FFFFFF',
        primary: '#333',
        secondary: '#F0F0F0',
        accent: '#333',
        text: '#000000',
        lightText: '#666666',
        border: '#CCCCCC'
    };

    const styles = {
        container: `
            font-family: 'PT Sans', sans-serif;
            background-color: ${colors.background};
            color: ${colors.text};
            padding: 20px;
            border-radius: 12px;
            width: 400px;
            border: 1px solid ${colors.border};
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
        `,
        topContainer: `
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        `,
        column: `
            display: flex;
            flex-direction: column;
            width: 48%;
            justify-content: space-between;
        `,
        select: `
            padding: 8px;
            margin: 5px 0;
            border: 1px solid ${colors.secondary};
            border-radius: 6px;
            background-color: ${colors.background};
            color: ${colors.text};
            font-size: 14px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
            height: 36px; 
        `,
        bpmContainer: `
            display: flex;
            flex-direction: column;
            //align-items: center;
            width: 100%;  
        `,
        bpmLabel: `
            font-size: 14px;
            margin-bottom: 5px;
            color: ${colors.text};
        `,
        bpmInput: `
            padding: 8px;
            border: 1px solid ${colors.secondary};
            border-radius: 6px;
            background-color: ${colors.background};
            color: ${colors.text};
            font-size: 14px;
            text-align: center;
            width: 100%; 
            height: 36px;
        `,
        timelineContainer: `
            position: relative;
            width: 100%;
            margin: 20px 0;
            display: flex;
            align-items: center;
        `,
        timeline: `
            flex-grow: 1;
            -webkit-appearance: none;
            background: ${colors.secondary};
            outline: none;
            border-radius: 15px;
            overflow: visible;
            height: 8px;
        `,
        button: `
            width: 40px;
            height: 40px;
            padding: 10px;
            border: none;
            border-radius: 50%;
            background-color: ${colors.primary};
            color: ${colors.background};
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0px 10px 0px 10px;
        `,
        timeDisplay: `
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: ${colors.lightText};
            margin: 0px 0px 0px 10px;
        `,
        buttonContainer: `
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
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
            display: flex;
            align-items: center;
            justify-content: center;
        `,
        buttonIcon: `
            margin-right: 5px;
        `
    };

    const createSVG = (svgString) => {
        const div = document.createElement('div');
        div.innerHTML = svgString.trim();
        return div.firstChild;
    };

    const playSVG = createSVG(`
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-circle-play"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
    `);

    const pauseSVG = createSVG(`
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-circle-pause"><circle cx="12" cy="12" r="10"/><line x1="10" x2="10" y1="15" y2="9"/><line x1="14" x2="14" y1="15" y2="9"/></svg>
    `);

    const midiSVG = createSVG(`
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-keyboard-music"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="M6 8h4"/><path d="M14 8h.01"/><path d="M18 8h.01"/><path d="M2 12h20"/><path d="M6 12v4"/><path d="M10 12v4"/><path d="M14 12v4"/><path d="M18 12v4"/></svg>        
    `);

    const wavSVG = createSVG(`
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-audio-lines"><path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/></svg>     
    `);

    const createElementWithStyle = (tag, style) => {
        const element = document.createElement(tag);
        element.style.cssText = style;
        return element;
    };

    const container = createElementWithStyle('div', styles.container);

// Barre de lecture et bouton Play
const timelineContainer = createElementWithStyle('div', styles.timelineContainer);
const timelineSlider = createElementWithStyle('input', styles.timeline);
Object.assign(timelineSlider, { type: 'range', min: 0, max: 100, value: 0 });

const playButton = createElementWithStyle('button', styles.button);
playButton.appendChild(playSVG);

timelineContainer.append(timelineSlider, playButton);

const timeDisplay = createElementWithStyle('div', styles.timeDisplay);
const currentTime = document.createElement('span');
currentTime.textContent = '0:00';
const totalTime = document.createElement('span');
totalTime.textContent = '0:00';
timeDisplay.append(currentTime, totalTime);

// Colonne de gauche pour les instruments
const leftColumn = createElementWithStyle('div', styles.column);
const instrumentsContainer = createElementWithStyle('div', styles.instrumentsContainer);
leftColumn.appendChild(instrumentsContainer);

// Colonne de droite pour le BPM
const rightColumn = createElementWithStyle('div', styles.column);
const bpmContainer = createElementWithStyle('div', styles.bpmContainer);
const bpmLabel = createElementWithStyle('label', styles.bpmLabel);
bpmLabel.textContent = 'Tempo';
const bpmInput = createElementWithStyle('input', styles.bpmInput);
Object.assign(bpmInput, { type: 'number', min: 60, max: 240, value: model.get('tempo') });
bpmContainer.append(bpmLabel, bpmInput);
rightColumn.appendChild(bpmContainer);

// Ajout des éléments au conteneur principal
const topContainer = createElementWithStyle('div', styles.topContainer);
topContainer.append(leftColumn, rightColumn);

container.append(topContainer, timelineContainer, timeDisplay);

    const showDebug = model.get('show_debug');
    const debugDiv = document.createElement('div');

    const updateDebug = msg => {
        if (showDebug) {
            model.set('debug_msg', msg);
            model.save_changes();
            debugDiv.textContent = msg;
            console.log(msg);
        }
    };

    updateDebug('Starting widget initialization...');

    // Load Tone.js and @tonejs/midi dynamically

    let Tone, MidiModule;
    try {
        updateDebug('Loading Tone.js...');
        await new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js';
            script.onload = () => {
                Tone = window.Tone;
                resolve();
            };
            script.onerror = reject;
            document.head.appendChild(script);
        });
        updateDebug('Tone.js loaded successfully');

        updateDebug('Loading @tonejs/midi...');
        MidiModule = await import('https://unpkg.com/@tonejs/midi@2.0.28/build/Midi.js');
        updateDebug('@tonejs/midi loaded successfully');

        if (!Tone || !Tone.Transport) {
            throw new Error('Tone.js or Tone.Transport is not available');
        }

        updateDebug('Tone and Midi objects are ready');

        // Initialize audio context
        await Tone.start();
        updateDebug('Audio context started');

        const customInstruments = model.get('custom_instruments');
        const synthTypes = [...Object.keys(customInstruments), 'Synth', 'AMSynth', 'DuoSynth', 'FMSynth', 'MembraneSynth', 'MetalSynth', 'MonoSynth', 'PluckSynth', 'PolySynth'];
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
            instrumentsContainer.appendChild(synthSelectorItem);  // Changé de synthSelectors à instrumentsContainer
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
                    time: time * 60 / Tone.Transport.bpm.value,  // Convert time to seconds based on BPM
                    noteName: midi === null ? null :
                        (Array.isArray(midi) 
                            ? midi.map(m => m !== null ? Tone.Frequency(m, "midi").toNote() : null)
                            : Tone.Frequency(midi, "midi").toNote()),
                    duration: duration * 60 / Tone.Transport.bpm.value  // Convert duration to seconds based on BPM
                }))).start(0);
                parts.push(part);
            });

            totalDuration = Math.max(...musicData.flat().map(note => note[2] + note[1])) * 60 / Tone.Transport.bpm.value;
            Tone.Transport.loopEnd = totalDuration;
            Tone.Transport.loop = true;

            totalTime.textContent = formatTime(totalDuration);
            updateDebug(`Audio initialized, duration: ${totalDuration.toFixed(2)}s, BPM: ${Tone.Transport.bpm.value}`);
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
                if (!Tone || !Tone.Transport) {
                    throw new Error('Tone.js or Tone.Transport is not available');
                }
                if (Tone.Transport.state === 'started') {
                    await Tone.Transport.stop();
                    playButton.innerHTML = '';
                    playButton.appendChild(playSVG.cloneNode(true));
                    updateDebug('Playback stopped');
                } else {
                    if (synths.length === 0) initAudio();
                    await Tone.Transport.start();
                    playButton.innerHTML = '';
                    playButton.appendChild(pauseSVG.cloneNode(true));
                    updateDebug(`Playback started, BPM: ${Tone.Transport.bpm.value}`);
                    updateTimeline();
                }
            } catch (error) {
                updateDebug(`Error during playback: ${error.message}`);
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

        // download midi button
        const buttonContainer = createElementWithStyle('div', styles.buttonContainer);
        const downloadMIDIButton = createElementWithStyle('button', styles.downloadButton);
        const midiIcon = midiSVG.cloneNode(true);
        midiIcon.style.cssText = styles.buttonIcon;
        const midiText = document.createElement('span');
        midiText.textContent = 'MIDI';
        downloadMIDIButton.appendChild(midiIcon);
        downloadMIDIButton.appendChild(midiText);

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
                                track.addNote({
                                    midi: note,
                                    time: time * 60 / bpm,
                                    duration: duration * 60 / bpm
                                });
                            }
                        });
                    } else {
                        // C'est une note simple
                        track.addNote({
                            midi: midiNote,
                            time: time * 60 / bpm,
                            duration: duration * 60 / bpm
                        });
                    }
                });
            });
            midi.header.setTempo(bpm);
            return midi.toArray();
        };

        downloadMIDIButton.onclick = () => {
            try {
                const midiData = generateMIDI();
                const blob = new Blob([midiData], { type: "audio/midi" });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'generated_music.mid';
                link.click();
                URL.revokeObjectURL(url);
                console.log('MIDI file downloaded');
            } catch (error) {
                console.error('Error generating MIDI:', error);
            }
        };

        const downloadWavButton = createElementWithStyle('button', styles.downloadButton);
        const wavIcon = wavSVG.cloneNode(true);
        wavIcon.style.cssText = styles.buttonIcon;
        const wavText = document.createElement('span');
        wavText.textContent = 'WAV';
        downloadWavButton.appendChild(wavIcon);
        downloadWavButton.appendChild(wavText);

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
            const bpm = Tone.Transport.bpm.value;
            const duration = Math.max(...musicData.flat().map(note => note[2] + note[1])) * 60 / bpm;

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
                        time: time * 60 / bpm,
                        noteName: midi === null ? null :
                            (Array.isArray(midi) 
                                ? midi.map(m => m !== null ? Tone.Frequency(m, "midi").toNote() : null)
                                : Tone.Frequency(midi, "midi").toNote()),
                        duration: duration * 60 / bpm
                    }))).start(0);
                });

                transport.bpm.value = bpm;
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
 
        timelineContainer.append(timelineSlider, timeDisplay);
        container.appendChild(timelineContainer);

        buttonContainer.append(downloadMIDIButton, downloadWavButton);
        container.appendChild(buttonContainer);

        if (showDebug) {
            container.appendChild(debugDiv);
        }
        el.appendChild(container);

        // Set initial BPM
        updateDebug(`Setting initial BPM to ${model.get('tempo')}`);
        Tone.Transport.bpm.value = model.get('tempo');
        updateDebug(`BPM set to ${Tone.Transport.bpm.value}`);

        updateDebug('Widget initialization completed successfully');

        // Cleanup function
        return () => {
            if (Tone && Tone.Transport) {
                Tone.Transport.stop();
                Tone.Transport.cancel();
            }
            if (synths) {
                synths.forEach(s => s.dispose());
            }
            if (parts) {
                parts.forEach(p => p.dispose());
            }
        };
    } catch (error) {
        updateDebug(`Error initializing widget: ${error.message}`);
        console.error('Error initializing widget:', error);
    }
}


export default { render };
    """

@staticmethod
def show(
    tracks: Union[List[Tuple[Union[int, List[int], None], float, float]], List[List[Tuple[Union[int, List[int], None], float, float]]]],
    custom_instruments: Dict[str, Dict] = None,
    tempo: int = 120,
    show_debug: bool = False  # Add this parameter
) -> anywidget.AnyWidget:
    """
    Shows a music player widget for the given tracks.

    Args:
        tracks: A list of tracks, where each track is a list of tuples. Each tuple contains:
            - The MIDI note number or a list of MIDI note numbers (or None for silence)
            - The duration of the note in beats
            - The start time of the note in beats
        custom_instruments: A dictionary mapping instrument names to custom Tone.js synth configurations.
        tempo: The initial BPM (beats per minute) for the player.
        show_debug: A boolean indicating whether to show debug messages in the widget.

    Returns:
        A MusicPlayer widget.
    """
    if not isinstance(tracks[0], list):
        tracks = [tracks]

    custom_instruments = custom_instruments or {}

    return (
        MusicPlayer(
            tracks=tracks,
            custom_instruments=custom_instruments,
            tempo=tempo,
            show_debug=show_debug,
            debug_msg="Widget initialized",
        )
    )