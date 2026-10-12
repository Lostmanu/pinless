"""python -m pinless: the same as the pinless command (when it is not on the PATH)."""
from .cli import main

raise SystemExit(main())
