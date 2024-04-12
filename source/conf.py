# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Djalgo'
copyright = '2024, Essi Parent'
author = 'Essi Parent'
release = '0.1-alpha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
    'nbsphinx',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
# html_css_files = ['custom.css']
html_logo = '_images/logo.png'
html_sidebars = {
   '**': [
       'globaltoc.html',  # Includes the global TOC; adapt as necessary
       # 'relations.html',  # Provides the Previous / Next links
       'searchbox.html',  # Includes the search box
       'custom_sidebar.html'
   ]
}
