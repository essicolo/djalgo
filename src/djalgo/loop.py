import numpy as np
from . import utils

import plotly.graph_objects as go
import colorsys

class Polyloop:
    """
    Represents a collection of polyloops, which are sequences of musical notes.
    """

    def __init__(self, polyloops, measure_length=4, insert_rests=True):
        """
        Initializes a Polyloop object.

        Parameters:
        - polyloops (list): A list of polyloops. Each polyloop is expected to be in the form [(offset, pitch, duration), ...].
        - measure_length (int): The length of a measure in beats. Defaults to 4.
        - insert_rests (bool): Whether to insert rests in the polyloops. Defaults to True.
        """
        self.measure_length = measure_length
        self.polyloops = [utils.fill_gaps_with_rests(polyloop) for polyloop in polyloops] if insert_rests else polyloops

    def plot_polyloops(self, pulse=1/4, colors=None):
        """
        Plots the given polyloops as a radar chart, including arcs to represent the duration of each note.

        Parameters:
        - pulse (float): The duration of each pulse in beats. Defaults to 1/4.
        - colors (list): A list of colors to use for the plot. If not provided, a default color scheme will be used.

        Returns:
        - fig (plotly.graph_objects.Figure): The generated radar chart figure.
        """
        self.polyloops = [self.polyloops] if not any(isinstance(i, list) for i in self.polyloops) else self.polyloops
        polyloops_without_rests = [[note for note in polyloop if note[0] is not None] for polyloop in self.polyloops]

        n_polyloops = len(self.polyloops)
        traces = []
        #colors = go.Figure().layout.template.layout.colorway if colors is None else colors
        if colors is None:
            colors = [colorsys.hsv_to_rgb(i/n_polyloops, 1, 1) for i in range(n_polyloops)]
            #colors = [colorsys.hls_to_rgb(0, i/n_polyloops, 0) for i in range(n_polyloops)]
            colors = ['rgba(%d, %d, %d, 0.5)' % (int(r*255), int(g*255), int(b*255)) for r, g, b in colors]

        fig = go.Figure()

        for i, polyloop in enumerate(polyloops_without_rests):
            for _, duration, offset in polyloop:  # Ignore the pitch component
                start_theta, duration_theta = offset * 360 / self.measure_length, duration * 360 / self.measure_length
                arc = np.linspace(start_theta, start_theta + duration_theta, 100)  # Generate points for a smooth arc
                r = [n_polyloops-i-1] * 100  # Constant radius for the arc
                
                fig.add_trace(go.Scatterpolar(
                    r=r,
                    theta=arc % 360,  # Ensure theta is within 0-360 range
                    mode='lines',
                    line=dict(color='rgba(60, 60, 60, 0.65)', width=8), # colors[i % len(colors)]
                    name=f'Polyloop {i+1} Duration',
                    showlegend=False
                ))

            for _, duration, offset in polyloop:
                start_theta, end_theta = offset * 360 / self.measure_length, (offset + duration) * 360 / self.measure_length
                for theta in [start_theta, end_theta]:
                    fig.add_trace(go.Scatterpolar(
                        r=[n_polyloops-i-0.9, n_polyloops-i-1.1],
                        theta=[theta % 360, theta % 360],
                        mode='lines',
                        line=dict(color='Black', width=3),
                        name=f'Polyloop {i+1} Start/End',
                        showlegend=False
                    ))

            if polyloop:
                start_thetas = [offset * 360 / self.measure_length for _, _, offset in polyloop]
                start_thetas.append(start_thetas[0])

            traces.append(go.Scatterpolar(
                r=[n_polyloops-i-1]*(len(polyloop)+1),  # Account for the loop closure
                theta=start_thetas,
                mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.65)', width=1),
                fill='toself',
                fillcolor=colors[i],
                name=f'Polyloop {i}',
                showlegend=False
            ))

        for trace in reversed(traces):
            fig.add_trace(trace)
        
        tickvals = np.linspace(0, 360, int(self.measure_length/pulse), endpoint=False)
        ticktext = [str(i % self.measure_length) for i in np.arange(0, self.measure_length, pulse)]
        radial_tickvals = np.arange(0, n_polyloops, 1)

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[n_polyloops, -0.1],
                    tickvals=radial_tickvals,
                    ticktext=[str(i) for i in radial_tickvals]
                ),
                angularaxis=dict(
                    tickvals=tickvals,
                    ticktext=ticktext,
                    direction="clockwise",
                    rotation=90
                )
            ),
            template='none',
            showlegend=True
        )

        fig.add_annotation(
            x=0.5, y=0.5, text="↻", showarrow=False, font=dict(size=30, color='White'),
            xref="paper", yref="paper"
        )

        return fig
