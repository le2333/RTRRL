"""Implementations kept to be checked against, and never to be run from.

Nothing that ships imports anything here. That used to be a sentence in a
docstring -- ``upstream_stream_ac.py`` sat among the real kernels and asked to be
left alone -- and it is a directory now so that the rule is enforced by where a
file is rather than by whether somebody read it. This package is outside
``memorax``, so a released import cannot reach it even by accident; the tests
find it because ``pytest.ini`` puts the checkout root on the path.

What belongs here is anything that answers the question "is this the same
arithmetic": a fork's own copy of a kernel, a published algorithm's wiring
re-hosted on this side's networks, a component's published counterpart. A
whole-program reference and a single cell's reference are the same kind of
thing and live together.

The repository has already made the choice these files embody. Golden
snapshots of the numbers were carried once and deleted, and the byte-identical
upstream kernel came back in their place: a recorded number can say that
something changed, and only a live reference can say what it changed into and
whether the change was right. A reference that stops compiling against its
dependencies is also telling you something, which a stored array cannot.
"""
