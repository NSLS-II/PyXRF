import logging
logger = logging.getLogger(__name__)

import enaml
with enaml.imports():
    from .main_window import XRFGui
