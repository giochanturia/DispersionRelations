# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Dispersion Relations"
copyright = "2025, George Chanturia"
author = "George Chanturia"
release = "1.0.0"
html_title = f"{project} ({release})"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",  # Equations
    "sphinxcontrib.bibtex",  # Citations
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_rtd_theme",
    "sphinx_copybutton",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Sorting the objects
autodoc_member_order = "bysource"

todo_include_todos = True

# -- Options for plotting extension
plot_include_source = False
plot_html_show_formats = False
plot_html_show_source_link = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_theme_options = {
    "logo": {
        "image_light": "_static/DR_light.png",
        "image_dark": "_static/DR_dark.png",
    },
    "github_url": "https://github.com/giochanturia/DispersionRelations",
    "show_nav_level": 2,
}
html_favicon = "_static/DR_logo.png"
html_show_sourcelink = False
html_show_sphinx = False

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

numfig = True

bibtex_bibfiles = ["literature.bib"]


from docutils import nodes
from docutils.parsers.rst import roles


def red_color_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.raw("", f'<span style="color:red">{text}</span>', format="html")
    return [node], []


def bold_red_color_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node = nodes.raw(
        "", f'<strong><span style="color:red">{text}</span></strong>', format="html"
    )
    return [node], []


roles.register_local_role("red", red_color_role)
roles.register_local_role("boldred", bold_red_color_role)
