def setup_pluto():
    return {
        "command": [
            "julia",
            "--startup-file=no",
            "-e",
            'import Pluto; Pluto.run(host="127.0.0.1", port={port}, launch_browser=false, '
            "require_secret_for_open_links=false, require_secret_for_access=false)",
        ],
        "timeout": 300,
        "new_browser_tab": True,
        "launcher_entry": {"title": "Pluto"},
    }
