from setuptools import setup, find_packages
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# README.md가 있으면 PyPI 설명용으로 사용 (없어도 동작하도록 처리)
readme_path = BASE_DIR / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# requirements.txt를 그대로 install_requires로 쓰고 싶으면 아래 사용
req_path = BASE_DIR / "requirements.txt"
install_requires = []
if req_path.exists():
    install_requires = [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="autocoin",
    version="0.1.0",
    description="AutoCoin RL project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.12",

    packages=find_packages(include=["AgentSelling*", "HateSelling*","src*"]),

    include_package_data=True,
    install_requires=install_requires,
)
