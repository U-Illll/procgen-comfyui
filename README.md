# Procedural Image Pipeline with ComfyUI

A procedural rendering and image-refinement pipeline with optional ComfyUI integration over its local HTTP API. The repository contains pipeline code only; it does not bundle ComfyUI, model checkpoints, generated images, local datasets, or credentials. Some specialized validation helpers still use legacy output filenames and require manual review before use; the core pipeline and ComfyUI API adapter are the primary entry points.

## Scope

- Procedural scene description and raster/G-buffer stages.
- Optional ControlNet-based refinement through a locally configured ComfyUI instance.
- Material/reference acquisition helpers that write downloaded data to a user-selected output directory.

## Requirements

Python 3.11+ and the packages listed below. For GPU rendering, install and configure ComfyUI separately and use models whose licenses permit your intended use. The legacy Chrome-rendering path expects a `chromium` executable on `PATH` by default; override it with `PROCGEN_CHROME`.

## Configuration

Set output paths explicitly with command-line flags where supported. Configure the local ComfyUI endpoint with the `COMFYUI_URL` environment variable (default: `http://127.0.0.1:8188`) and its install directory with `COMFYUI_DIR` (default: `~/ComfyUI`). The renderer calls only that configured ComfyUI API endpoint.

## Data and privacy

Generated images, reference photos, material assets, workflow histories, access tokens, cookies, and personal or organization data are not included. Do not commit downloaded datasets or generated outputs.

## License

License to be selected after authorship and third-party dependency review.
