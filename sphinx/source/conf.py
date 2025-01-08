import datetime
import doctest

import sphinx_rtd_theme
import ggfm

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

exclude_patterns = []

autosummary_generate = True
templates_path = ['_templates']

project = 'GGFM'
author = 'BUPT-GAMMA LAB'
release = '0.1'
copyright = f'{datetime.datetime.now().year}, {author}'

# version = ggfm.__version__
# release = ggfm.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration



language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
