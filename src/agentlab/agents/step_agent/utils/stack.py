from dataclasses import dataclass

from ..prompt_agent import PromptAgent


@dataclass
class Element:
    agent: PromptAgent
    objective: str


class Stack:
    """Adapted from https://github.com/asappresearch/webagents-step/blob/main/src/webagents_step/utils/stack.py"""
    def __init__(self) -> None:
        self.items: list[Element] = []
    
    def size(self) -> int:
        return len(self.items)

    def is_empty(self):
        return len(self.items) == 0
    
    def push(self, item: Element):
        self.items.append(item)
    
    def pop(self) -> Element:
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self) -> Element:
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]
