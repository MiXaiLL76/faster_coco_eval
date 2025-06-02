import os
import sys
from datetime import datetime

from faster_coco_eval import __author__, __version__

sys.path.insert(0, os.path.abspath('..'))

project = f'faster-coco-eval {__version__}'
current_year = datetime.now().year
copyright = f'2024-{current_year}, {__author__}'
author = __author__
release = __version__
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "nbsphinx"
]

exclude_patterns = []
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def setup(app):
    app.add_css_file('custom_nbsphinx.css')
