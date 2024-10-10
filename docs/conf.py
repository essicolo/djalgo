import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'djalgo'
copyright = '2024, Essi Parent'
author = 'Essi Parent'
release = "0.2.2"

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'recommonmark',
    'nbsphinx',
    'jupyter_sphinx',
]

# General configuration
master_doc = 'index'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
    '.ipynb': 'jupyter_notebook',
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'furo'
html_static_path = ['_static']
templates_path = ['_templates']

# Nbsphinx settings
nbsphinx_execute = 'never'
html_logo = '_static/logo.png'
html_js_files = [
    'https://cdn.jsdelivr.net/npm/abcjs@6.2.2/dist/abcjs-basic-min.js',
]

# Jupyter widgets configuration
jupyter_sphinx_require_url = 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js'

def setup(app):
    app.add_css_file('https://cdn.jsdelivr.net/npm/abcjs@6.2.2/abcjs-audio.css')
    app.add_js_file('https://cdn.jsdelivr.net/npm/abcjs@6.2.2/dist/abcjs-basic-min.js')
