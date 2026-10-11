import setuptools

setuptools.setup(
    name="pluto-launcher",
    version="1.0.0",
    description="Starts Pluto.jl behind jupyter-server-proxy (used on Binder)",
    packages=["pluto_launcher"],
    install_requires=["jupyter-server-proxy"],
    entry_points={
        "jupyter_serverproxy_servers": [
            "pluto = pluto_launcher:setup_pluto",
        ]
    },
)
