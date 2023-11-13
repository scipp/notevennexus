# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass

from .tree import Dataset, Group


@dataclass
class Violation:
    name: str
    description: str | None = None

    def format(self) -> str:
        return (
            f"{self.name} ({self.description})"
            if self.description is not None
            else self.name
        )


class Validator:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._count = 0
        self._violations: list[Violation] = []

    def applies_to(self, node: Dataset | Group) -> bool:
        raise NotImplementedError

    def validate(
        self, tree: dict[str, Dataset | Group], node: Dataset | Group
    ) -> Violation | None:
        raise NotImplementedError

    def apply(self, tree: dict[str, Dataset | Group], node: Dataset | Group) -> None:
        if self.applies_to(node):
            self._count += 1
            if (violation := self.validate(tree, node)) is not None:
                self._violations.append(violation)

    def violations(self) -> list[Violation]:
        return self._violations

    def report(self) -> str:
        content = '\n    '.join([v.format() for v in self.violations()])
        violations = (
            f'Violations ({len(self._violations)}/{self._count}):\n    {content}'
        )
        return f"{self.name}: {self.description}\n{violations}"


def validate(tree: dict[str, Dataset | Group], validators: list[Validator]) -> None:
    for node in tree.values():
        for validator in validators:
            validator.apply(tree, node)
