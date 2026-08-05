"""What every registered module owes, whatever it builds.

A registered module is one class that both declares and implements: ``param()``
says what an experiment may choose, ``build()`` makes the thing. Nothing here
names a component. Whatever is registered under ``memorax.component`` and
``memorax.algorithm`` is swept, so a module added tomorrow is covered the moment
it is registered rather than when someone remembers to list it.

Three rules carry most of the weight.

**Declaring is opting in.** ``param()`` lists only what an experiment may
choose. A knob that is fixed by the implementation, and a value the graph
supplies, are both simply absent -- so absence is the answer to "may this be
searched", and nothing has to be marked "not a parameter".

**Declaration and use cannot drift.** They are on one class, so a name declared
and never read, and a name read and never declared, are both visible from here:
``build`` is driven with a mapping that remembers every lookup.

**A declaration describes one module.** It never reaches into whatever it
builds. Nesting is a later question; until then a declaration is flat.
"""

from __future__ import annotations

import inspect
import pkgutil
from collections.abc import Iterator, Mapping
from importlib import import_module
from typing import Any

import pytest

from memorax import algorithm as algorithm_package
from memorax import component as component_package
from memorax.modules import (
    BuildContext,
    ChoiceParameter,
    FloatParameter,
    IntParameter,
)

LEAVES = (ChoiceParameter, FloatParameter, IntParameter)

# One width, so that a module which reads it has something to read. Its value is
# nothing: what is asserted is where it comes from, not what it is.
CONTEXT = BuildContext(features=8)


def registered() -> dict[str, type]:
    """Every explicitly registered class, by its canonical name.

    Three levels, as the design has them: directory, then file, then the classes
    that file lists. A class the file does not list is an implementation detail
    and is not swept.
    """

    found: dict[str, type] = {}
    for directory, package in (
        ("algorithm", algorithm_package),
        ("component", component_package),
    ):
        for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
            if info.ispkg:
                continue
            imported = import_module(f"{package.__name__}.{info.name}")
            for one in getattr(imported, "MODULES", ()):
                found[f"{directory}.{info.name}.{one.__name__}"] = one
    return found


MODULES = registered()


def declaration(module: type) -> Mapping[str, Any]:
    return module.param()


class Watched(Mapping):
    """A parameter mapping that remembers which names were looked up."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)
        self.read: set[str] = set()

    def __getitem__(self, name: str) -> Any:
        self.read.add(name)
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def one_value(leaf) -> Any:
    """A value inside the leaf's own search domain, for driving ``build``."""

    if isinstance(leaf, ChoiceParameter):
        return leaf.search[0]
    return leaf.search[0]


def inside(bounds, value) -> bool:
    low, high = bounds
    return (low is None or value >= low) and (high is None or value <= high)


def test_there_are_registered_modules_to_check():
    """An empty sweep would pass every other test in this file."""

    assert MODULES, (
        "no class is registered under memorax.component or memorax.algorithm; "
        "a file registers by listing its classes in MODULES"
    )


@pytest.fixture(params=sorted(MODULES), ids=sorted(MODULES))
def module(request) -> type:
    return MODULES[request.param]


def test_a_module_declares_and_builds_without_being_instantiated(module):
    """Both are class methods, because a catalog reads a declaration off a class.

    The catalog is built from what a file registers, not from anything the file
    can run. Requiring an instance would mean constructing one before knowing
    what it accepts, which is the wrong way round.
    """

    for name in ("param", "build"):
        found = inspect.getattr_static(module, name, None)
        assert isinstance(found, classmethod), f"{name} must be a class method"


def test_a_declaration_is_flat(module):
    """One module, one level. What it builds declares its own.

    A declaration that reached into a child would put the child's domain in two
    places, and the two would not have to agree.
    """

    for name, leaf in declaration(module).items():
        assert isinstance(leaf, LEAVES), f"{name} is {type(leaf).__name__}"


def test_a_declaration_carries_no_default_value(module):
    """Only the two domains: what is legal, and what is searched by default.

    A single default would be a third thing, and no run would use it: an
    experiment either pins a value or leaves the domain to be searched.
    """

    for name, leaf in declaration(module).items():
        assert not hasattr(leaf, "placeholder"), name
        assert not hasattr(leaf, "default"), name


def test_the_search_domain_is_inside_the_legal_one(module):
    """The default search range cannot ask for what the module refuses."""

    for name, leaf in declaration(module).items():
        if isinstance(leaf, ChoiceParameter):
            assert set(leaf.search) <= set(leaf.valid), name
            assert leaf.search, name
            continue
        low, high = leaf.search
        assert low <= high, name
        assert inside(leaf.valid, low), name
        assert inside(leaf.valid, high), name


def test_a_module_reads_every_name_it_declares(module):
    """Declared and never read is a knob an experiment can turn to no effect."""

    params = Watched({name: one_value(leaf) for name, leaf in declaration(module).items()})
    module.build(params=params, context=CONTEXT)

    assert not set(declaration(module)) - params.read


def test_a_module_declares_every_name_it_reads(module):
    """Read and never declared arrives as a KeyError once a job has started."""

    params = Watched({name: one_value(leaf) for name, leaf in declaration(module).items()})
    module.build(params=params, context=CONTEXT)

    assert not params.read - set(declaration(module))


def test_what_the_graph_supplies_is_not_declared(module):
    """The width a module is handed is the sequence's, not an experiment's.

    It reaches ``build`` through the context, so declaring it too would let an
    experiment name a value the graph then overrules.
    """

    supplied = {field for field in vars(CONTEXT) if not field.startswith("_")}

    assert not supplied & set(declaration(module))
