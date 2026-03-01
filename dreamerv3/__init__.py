import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

# from .agent_teacher import Agent
from .agent_teacher import Agent as agent_teacher
from .agent_student import Agent as agent_student
from .agent_student_bisim import Agent as agent_student_bisim
from .agent_policy_distillation import Agent as agent_policy_distillation
from .agent_vlm import Agent as agent_vlm