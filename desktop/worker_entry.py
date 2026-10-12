"""PyInstaller entry point (absolute import also works in a frozen build)."""
from rst19.desktop_worker import main

raise SystemExit(main())
