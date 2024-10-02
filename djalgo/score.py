import anywidget
import traitlets
from typing import List, Union, Tuple
from .conversion import to_abc

class ABCjsWidget(anywidget.AnyWidget):
    abc = traitlets.Unicode().tag(sync=True)
    _esm = """
    import * as abcjs from "https://cdn.jsdelivr.net/npm/abcjs@6.2.2/+esm";
    export function render({ model, el }) {
        function updateABC() {
            el.innerHTML = '';
            abcjs.renderAbc(el, model.get('abc'));
        }
        model.on('change:abc', updateABC);
        updateABC();
    }
    export default { render };
    """

@staticmethod
def show(tracks: Union[List[Tuple[int, float, float]], List[List[Tuple[int, float, float]]]],
         key: str = "C",
         clef: str = "treble",
         time_signature: str = "4/4",
         title: str = None,
         tempo: int = 120
    ) -> ABCjsWidget:
    if isinstance(tracks[0], tuple):
        tracks = [tracks]
    abc = to_abc(tracks, key, clef, time_signature, title, tempo)
    widget = ABCjsWidget(abc=abc)
    return widget
