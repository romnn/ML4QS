"""
Tasks for maintaining the project.
Execute 'invoke --list' for guidance on using Invoke
"""
import pprint
import shutil
from pathlib import Path

from invoke import task

Path().expanduser()

ROOT_DIR = Path(__file__).parent
PYTHON_DIRS = [str(d) for d in [ROOT_DIR]]


def _delete_file(file):
    try:
        file.unlink(missing_ok=True)
    except TypeError:
        # missing_ok argument added in 3.8
        try:
            file.unlink()
        except FileNotFoundError:
            pass


@task(help={"check": "Checks if source is formatted without applying changes"})
def format(c, check=False):
    """Format code"""
    python_dirs_string = " ".join(PYTHON_DIRS)
    black_options = "--diff" if check else ""
    c.run("pipenv run black {} {}".format(black_options, python_dirs_string))
    isort_options = "{}".format("--check-only" if check else "")
    c.run("pipenv run isort {} {}".format(isort_options, python_dirs_string))


@task
def lint(c):
    """Lint code"""
    c.run("pipenv run flake8 {}".format(ROOT_DIR))


@task
def clean_build(c):
    """Clean up files from package building"""
    c.run("rm -fr .eggs/")
    c.run("find . -name '*.egg-info' -exec rm -fr {} +")
    c.run("find . -name '*.egg' -exec rm -f {} +")


@task
def clean_python(c):
    """Clean up python file artifacts"""
    c.run("find . -name '*.pyc' -exec rm -f {} +")
    c.run("find . -name '*.pyo' -exec rm -f {} +")
    c.run("find . -name '*~' -exec rm -f {} +")
    c.run("find . -name '__pycache__' -exec rm -fr {} +")


@task
def clean_tests(c):
    """Clean up files from testing"""
    _delete_file(COVERAGE_FILE)
    shutil.rmtree(TOX_DIR, ignore_errors=True)
    shutil.rmtree(COVERAGE_DIR, ignore_errors=True)


@task(pre=[clean_build, clean_python, clean_tests])
def clean(c):
    """Runs all clean sub-tasks"""
    pass
