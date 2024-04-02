# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
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


class Validator(ABC):
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def applies_to(self, node: Dataset | Group) -> bool:
        """Return True if this validator applies to the given node"""

    @abstractmethod
    def validate(self, node: Dataset | Group) -> Violation | None:
        """Return a Violation if the given node violates this validator"""


class ValidationResult:
    def __init__(self, validator: Validator) -> None:
        self.checks = 0
        self.fails = 0
        self.validator = validator
        self.violations: list[Violation] = []

    def apply(self, node: Dataset | Group) -> None:
        if self.validator.applies_to(node):
            self.checks += 1
            if (violation := self.validator.validate(node)) is not None:
                self.fails += 1
                self.violations.append(violation)

    def format_details(self) -> str:
        details = ''
        for violation in self.violations:
            details += f"{self.validator.name} @ {violation.format()}\n"
        return details

    def format_summary(self) -> str:
        return f"{self.validator.name}: {self.fails}/{self.checks}\n"


def validate(group: Group, validators: list[Validator]) -> dict[type, ValidationResult]:
    tree = unroll_tree(group)
    results = {type(v): ValidationResult(v) for v in validators}
    for node in tree.values():
        for validation in results.values():
            validation.apply(node)
    return results


def report(results: dict[type, ValidationResult]) -> str:
    details = 'Violations\n----------\n'
    summary = 'Summary\n-------\n'
    total_checks = 0
    total_violations = 0
    for result in results.values():
        total_checks += result.checks
        total_violations += result.fails
        details += result.format_details()
        summary += result.format_summary()
    summary += '\n'
    summary += f"Total: {total_violations}/{total_checks}"
    return f'{details}\n\n{summary}'


def has_violations(results: dict[type, ValidationResult]) -> bool:
    return any(result.fails for result in results.values())
