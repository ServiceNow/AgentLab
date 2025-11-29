from dataclasses import dataclass
import logging
from typing import Literal


@dataclass
class LlmRetrieverUtils:
    @staticmethod
    def remove_lines(tree: str, line_numbers: list[int]) -> str:
        """
        Remove all other lines that are not in line_numbers.
        Replace each part removed by the number of collapsed lines.

        Args:
            tree (str): The tree containing content to process
            line_numbers (list[int]): Line numbers to keep (1-indexed)

        Returns:
            str: Content with only specified lines kept, other parts replaced by tags
        """
        logging.info("Removing lines not in list..")
        lines = tree.splitlines()
        result_lines = []

        i = 0
        while i < len(lines):
            if i + 1 in line_numbers:  # line numbers are 1-indexed
                result_lines.append(lines[i])
                i += 1
            else:
                # Count consecutive lines to remove
                start = i
                while i < len(lines) and i + 1 not in line_numbers:
                    i += 1

                count = i - start
                # Get the indentation of the next line (if exists)
                # next_indentation = ""
                # if i < len(lines):
                #     next_indentation = lines[i][:-len(lines[i].lstrip())]

                # Create pruned tag with proper indentation
                tag = f"... pruned {count} lines ..."
                result_lines.append(tag)

        return "\n".join(result_lines)

    @staticmethod
    def remove_lines_keep_structure(
        tree: str, line_numbers: list[int], strategy: Literal["bid", "bid+role"]
    ) -> str:
        """
        Remove all other lines that are not in line_numbers.
        Keep the structure of the tree.

        Args:
            tree (str): The tree containing content to process
            line_numbers (list[int]): Line numbers to keep (1-indexed)
            strategy (Literal["bid", "bid+role"]): Strategy to keep structure
                - "bid": keep only the bid of the element and replace the rest of the line
                - "bid+role": keep the bid and role of the element and replace the rest of the line

        Returns:
            str: Content with only specified lines kept, other parts replaced by tags
        """
        logging.info("Removing lines not in list while keeping structure..")
        lines = tree.splitlines()
        result_lines = []

        for i, line in enumerate(lines):
            if i + 1 in line_numbers:
                result_lines.append(line)
            else:
                indentation = line[: len(line) - len(line.lstrip())]
                match strategy:
                    case "bid":
                        tag = (
                            line.split()[0] + " ... removed ..."
                        )  # keep bid and replace the rest of the line with removed
                    case "bid+role":
                        if ("[" not in line) and ("]" not in line):
                            tag = line.split()[
                                0
                            ]  # If the line does not contain a bid, keep only the role of the element
                        else:
                            tag = (
                                " ".join(line.split()[:2]) + " ... removed ..."
                            )  # keep bid and role of element and remove bid

                result_lines.append(f"{indentation}{tag}")

        return "\n".join(result_lines)
