# Configuration file for the Sphinx documentation builder.

import os
import sys
import tomli
sys.path.insert(0, os.path.abspath('../..'))  # Adjust this path to your project root

def get_version_from_pyproject():
    pyproject_path = os.path.join(os.path.dirname(__file__), '..', '..', 'pyproject.toml')
    try:
        with open(pyproject_path, 'rb') as f:
            pyproject_data = tomli.load(f)
        return pyproject_data['project']['version']
    except (FileNotFoundError, KeyError):
        print("Warning: Unable to read version from pyproject.toml")
        return "unknown"

# Project information
project = 'djalgo'
copyright = '2024, Essi Parent'
author = 'Essi Parent'
release = get_version_from_pyproject()

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx.ext.viewcode',  # Adds links to highlighted source code
    'sphinx.ext.intersphinx',  # Allows linking between different Sphinx projects
]

# General configuration
master_doc = 'index'
source_suffix = ['.rst', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'piccolo_theme'
htmlhelp_basename = 'djalgoDoc'
html_static_path = ['_static']
templates_path = ['_templates']
html_output_path = '../_build/html'
html_logo = '_static/logo.png'

# Sidebar configuration
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
        'custom_sidebar.html'
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
nbsphinx_execute = 'auto'

html_css_files = ["custom.css"]