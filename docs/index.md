:::{image} _static/logo.svg
:class: only-light
:alt: Chexus
:width: 60%
:align: center
:::

:::{image} _static/logo-dark.svg
:class: only-dark
:alt: Chexus
:width: 60%
:align: center
:::

#

<div style="display: block;width: 100%;font-size:1.2em;font-style:italic;color:#5a5a5a;text-align: center;">
    Validate and check NeXus files
    </br></br>
</div>

## Install

`````{tab-set}
````{tab-item} pip
```sh
pip install chexus
```
````
````{tab-item} conda
```sh
conda install -c conda-forge -c scipp chexus
```
````
`````

## Run

```bash
chexus <path-to-nexus-file>
```

This supports HDF5 as well as some JSON format.
There is also a Python API, but this is under construction and unstable.

## Options

- `--checksums`: Compute and print checksums.
- `--ignore-missing`: Skip the validators that have missing dependencies.
- `--exit-on-fail`: Return a non-zero exit code if validation fails.
- `-r`, `--root-path`: Path to the top-level group to validate. Default is `''`.

```{toctree}
---
hidden:
---

api-reference/index
developer/index
about/index
```
