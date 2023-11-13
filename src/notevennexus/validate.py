# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from dataclasses import dataclass

from .tree import Dataset, Group, unroll_tree


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

    @property
    def count(self) -> int:
        return self._count

    @property
    def violations(self) -> list[Violation]:
        return self._violations


def validate(group: Group, validators: list[Validator]) -> None:
    tree = unroll_tree(group)
    for node in tree.values():
        for validator in validators:
            validator.apply(tree, node)


def report(validators: list[Validator]) -> str:
    details = 'Violations\n----------\n'
    summary = 'Summary\n-------\n'
    total_checks = 0
    total_violations = 0
    for v in validators:
        total_checks += v._count
        total_violations += len(v.violations)
        for violation in v.violations:
            details += f"{v.name} @ {violation.format()}\n"
        summary += f"{v.name}: {len(v.violations)}/{v._count}\n"
    summary += f"Total: {total_violations}/{total_checks}"
    return f'{details}\n\n{summary}'
