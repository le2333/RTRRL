"""One file per recipe: a wiring, the space it can be searched over, a loop.

Nothing here is shared. Two entries that want the same agent with different
wiring are two files, because the alternative is a branch in a place that ends
up owning every recipe at once.
"""
