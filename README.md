# Core-CT
The `core-ct` scan library was built to assist geologists with the visualization and analysis of the CT scans of rock cores.

## Development Environment
1. Install `poetry` using the instructions at https://python-poetry.org/docs/#installation
2. Navigate to inside the repository 
3. Run `poetry install` to install the required dependencies
4. Use `poetry shell` to activate the newly created virtual environment

### Linting
To help maintain a clean codebase, this project makes use of the `ruff` linter. Due to settings in the GitHub repository, branches cannot be merged into `main` unless they pass linting checks. 

As such, this repository also supports the Python `pre-commit` library, which can be used to install a git pre-commit hook that requires the code to pass a linting check before it can be committed. These pre-commit hooks are defined in the `.pre-commit-config.yaml`.

To set up this linting hook perform the following actions:
1. Run `poetry install` to install `pre-commit` and `ruff`
2. Run `poetry run pre-commit install` to install the commit hooks

If you would prefer to manually run a linting check, simply use the following command.
```
poetry run ruff .
```

### Testing
Just as with linting, due to settings in the GitHub repository, branches cannot be merged into `main` unless they pass all tests as specified in the `tests` repository.

To run these tests simply use the following command:
```
poetry run pytest
```

## Resources
- https://github.com/mrsiegfried/SiegVent2023-Geology
- https://osu-mgr.org/sedct

## Credits
This software was built by the Golden Rocks team as a part of the Fall 2023 session of the Colorado School of Mines' [CSCI-370](https://cs-courses.mines.edu/csci370/index.html) course.

### Authors
Carla Ellefsen, Kira Hanson, Connor Sparks, Asa Sprow

### Sponsors
Zane Jobe, Ryan Venturelli
