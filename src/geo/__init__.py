"""Stage 5a — metric scale and georeferencing (§8.1).

``crs`` chooses the projected CRS and vertical datum; ``georef`` fits the reconstruction to
the filtered GPS (Stage 2 ``geo.txt``) and writes ``georef.json``, which every exporter reads.
"""
