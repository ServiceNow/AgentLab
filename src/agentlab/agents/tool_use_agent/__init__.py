import sys

from .tool_use_agent import *

# for backward compatibility of unpickling
sys.modules[__name__ + ".multi_tool_agent"] = sys.modules[__name__]
