import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'djalgo'
copyright = '2024, Essi Parent'
author = 'Essi Parent'
release = "0.2.3"

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'recommonmark'
]

# General configuration
master_doc = 'index'
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'furo'
html_static_path = ['_static']
templates_path = ['_templates']

html_logo = '_static/logo.png'
