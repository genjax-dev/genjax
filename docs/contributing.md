---
search:
  exclude: true
---

# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [Apache 2.0 license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

- [Source Code]
- [Documentation]
- [Issue Tracker]
- [Code of Conduct]

[apache 2.0 license]: https://opensource.org/licenses/Apache-2.0
[source code]: https://github.com/genjax-dev/genjax-chi
[documentation]: https://genjax.gen.dev
[issue tracker]: https://github.com/genjax-dev/genjax-chi/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of this project are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment
### Dev Container

The easiest way to get started is using the provided Dev Container configuration, which automatically sets up a complete development environment.

#### What gets installed?
- Python 3.11
- Poetry (dependency management)
- All project dependencies (including dev dependencies)
- Pre-commit hooks
- VS Code extensions for Python development, formatting, and testing

#### Setup

- **GitHub Codespaces**
  1. Click "Code" → "Codespaces" → "Codespace repository configuration" (next to "Create codespace on main")
  2. Click "New with options..."
  3. Choose `genjax` configuration (not `genjax-gpu` - Codespaces don't support GPU access)
  4. Wait for the container to build and dependencies to install (~5 minutes)
  5. Start coding - everything is ready to go

- **Local with VS Code:**
  1. Install [Docker](https://docs.docker.com/get-docker/) and [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
  2. Clone the repository
  3. Open in editor (VS Code/Cursor/Windsurf) and click "Reopen in Container" when prompted
  4. Choose either:
     - `base` - Standard development environment
     - `gpu` - Includes GPU access for CUDA workloads

  The setup process runs automatically and takes 2-3 minutes on first launch.

Note: Upon first startup, reload the window so all extensions can properly load now that setup is complete.

### Manual Setup
You need Python 3.7+ (we recommend 3.11+) and the following tools:

- [Poetry]
- [Nox]
- [nox-poetry]

Install the package with development requirements:

```console
$ poetry install
```

You can now run an interactive Python session:

```console
$ poetry run python
```

[poetry]: https://python-poetry.org/
[nox]: https://nox.thea.codes/
[nox-poetry]: https://nox-poetry.readthedocs.io/

## How to test the project

Run the full test suite:

```console
$ nox
```

List the available Nox sessions:

```console
$ nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ nox --session=tests
```

Unit tests are located in the _tests_ directory,
and are written using the [pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

- The Nox test suite must pass without errors and warnings.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though—we can always iterate on this.

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[pull request]: https://github.com/genjax-dev/genjax-chi/pulls
