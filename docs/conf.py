project = 'djalgo'
copyright = '2024, Essi Parent'
author = 'Essi Parent'
release = '0.3b'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

master_doc = 'index'
source_suffix = ['.rst', '.md']
htmlhelp_basename = 'djalgoDoc'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']
html_logo = '_images/logo.png'
html_sidebars = {
   '**': [
       'globaltoc.html',  # Includes the global TOC; adapt as necessary
       # 'relations.html',  # Provides the Previous / Next links
       'searchbox.html',  # Includes the search box
       'custom_sidebar.html'
   ]
}
