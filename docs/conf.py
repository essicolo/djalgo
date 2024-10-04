# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'djalgo'
copyright = '2024, Essi Parent'
author = 'Essi Parent'
release = "0.2"

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'recommonmark',
    'nbsphinx',
]

# General configuration
source_dir = 'source'
master_doc = 'index'
source_suffix = ['.rst', '.md', '.ipynb']
source_encoding = 'utf-8'
source_parsers = {'.md': 'recommonmark.parser.CommonMarkParser'}
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'furo'
htmlhelp_basename = 'djalgoDoc'
html_static_path = ['_static']
templates_path = ['_templates']
html_output_path = '_build/html'
html_logo = '_static/logo.png'

# Sidebar configuration
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# NBSphinx settings
nbsphinx_execute = 'never'