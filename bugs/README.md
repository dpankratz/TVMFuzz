## Bugs

In this directory are testcases for bugs found by TVMFuzz. They are designed such that they should crash or raise an assertion error to indicate the bug unless the bug has been patched. In this sense they can be considered regression tests.

The only exception is the directory `floormod` which was a special case where the operators `floormod` and `floordiv` produce a massive amount of IR and required a more thorough investigation into improving.