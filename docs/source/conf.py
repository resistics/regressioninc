# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import re
import regressioninc
from sphinx_gallery.sorting import FileNameSortKey

project = "Regression in C"
copyright = "2022, Neeraj Shah"
author = "Neeraj Shah"

# The full version, including alpha/beta/rc tags
release = regressioninc.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["setup.rst", "modules.rst"]

# regressionnc configuration
# autodoc
autosectionlabel_prefix_document = True
autodoc_member_order = "bysource"
autodoc_undoc_members = False
# napoleon extension
napoleon_numpy_docstring = True
napoleon_attr_annotations = False
# other configuration
plot_include_source = True
todo_include_todos = True
# intersphinx
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}
# copy button
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True
# pydantic configuration
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_config_member = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_hide_paramlist = True
autodoc_pydantic_field_show_default = True
# sphinx gallery
image_scrapers = "matplotlib"
sphinx_gallery_conf = {
    "run_stale_examples": True,
    "filename_pattern": f"{re.escape(os.sep)}eg_",
    "ignore_pattern": f"{re.escape(os.sep)}temp_",
    "remove_config_comments": True,
    "examples_dirs": ["../../examples"],
    "gallery_dirs": ["gallery"],
    "thumbnail_size": (300, 300),
    "image_scrapers": image_scrapers,
    "within_subsection_order": FileNameSortKey,
}

# code styles
pygments_style = "sphinx"
pygments_dark_style = "monokai"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_with_keys": True,
}
