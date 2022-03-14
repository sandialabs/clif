# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os

sys.path.append(os.path.abspath("."))
import doc

# -- Project information -----------------------------------------------------

project = "clif"
copyright = "2022, K. Chowdhary"
author = "K. Chowdhary"

# The full version, including alpha/beta/rc tags
release = "0.2.0"

# add path
sys.path.append(os.path.abspath("../../"))

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgconverter",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
]

todo_include_todos = True
numpydoc_class_members_toctree = True

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_short_title = "clif"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# pygments_style = "monokai"
# html_theme = 'sphinx_rtd_theme'

# book theme
html_theme = "sphinx_book_theme"
html_logo = "_static/clif_logo.png"
html_theme_options = {
    "collapse_navigation": False,
    "repository_url": "https://github.com/kennychowdhary/clif",
    "use_repository_button": True,
}
