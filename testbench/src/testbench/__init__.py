"""Judging whether two theoretically equivalent computations agree.

One stimulus, two implementations, probes where they correspond, and a verdict
at each probe. The vocabulary is here; what is being compared is not, and this
package never learns it.
"""

from testbench.gap import last_bits, relative

__all__ = ["last_bits", "relative"]
