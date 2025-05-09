"""Setup script for the env_nav_rl package."""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="env_nav_rl",
        packages=find_packages(include=["src", "src.*"]),
        package_dir={"": "."}
    ) 