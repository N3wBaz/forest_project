"""Nox sessions."""

# # import tempfile
# # from typing import Any

# # import nox
# # from nox.sessions import Session

# import tempfile
# from typing import (
#     Any,
# )
# import nox
# from nox.sessions import (
#     Session,
# )


# nox.options.sessions = ("black", "mypy", "flake8", "my_test")

# locations = "src", "noxfile.py"

import tempfile
from typing import Any

import nox
from nox.sessions import Session


nox.options.sessions = "black", "flake8", "mypy", "tests"
locations = "src", "noxfile.py"


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    """Install packages constrained by Poetry's lock file.
    By default newest versions of packages are installed,
    but we use versions from poetry.lock instead to guarantee
    reproducibility of sessions.
    """
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python="3.8")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python="3.8")
def flake8(session: Session) -> None:
    """Run flake8 code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "flake8")
    session.run("flake8", *args)


@nox.session(python="3.8")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


# @nox.session(python="3.8")
# def my_test(session: Session) -> None:
#     """Type-check using tests."""
#     args = session.posargs or locations
#     session.run("poetry", "install", "--no-dev", external=True)
#     install_with_constraints(session, "pytest")
#     session.run(
#         "pytest",
#         *args,
#     )

# @nox.session(python="3.8")
# def tests(session: Session) -> None:
#     """Run the test suite."""
#     args = session.posargs
#     session.run("poetry", "install", external=True)
#     install_with_constraints(session, "pytest")
#     session.run("pytest", *args)


# @nox.session(python="3.8")
# def tests(session: Session) -> None:
#     """Run the test suite."""
#     args = session.posargs
#     session.run("poetry", "install")

#     install_with_constraints(
#         session,
#         "pytest",
#         "click",
#     )
#     session.run("pytest", *args)


@nox.session(python="3.8")
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "run", "pytest")


# python = "^3.8poetry"
# pandas = "1.3.5"
# click = "^8.1.3"
# poetry = "^1.1.13"
# scikit-learn = "^1.0.2"
# pandas-profiling = "^3.2.0"
# mlflow = "^1.25.1"
# Boruta = "^0.3"
# virtualenv = "^20.14.1"


# [tool.poetry.dev-dependencies]
# pytest = "^6.2.5"
# black = "^22.3.0"
# flake8 = "^4.0.1"
# mypy = "^0.950"
# nox = "^2022.1.7"
