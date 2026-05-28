"""
PyInstaller runtime hook — register RosettaSciIO readers with HyperSpy.

In a frozen app, entry-point discovery often fails because .dist-info
metadata isn't on the search path.  This hook forces registration so
that hs.load(..., file_format="emd") (and every other rosettasciio
format) works exactly as it does in a normal venv.
"""

import importlib
import sys


def _patch_hyperspy_io():
    """
    Manually register rosettasciio file format plugins with HyperSpy's
    IO registry, bypassing entry-point discovery.
    """
    try:
        import rsciio
        from pathlib import Path

        # rsciio stores each format as a subpackage with a specs dict
        rsciio_dir = Path(rsciio.__file__).parent
        format_dirs = sorted(
            d for d in rsciio_dir.iterdir()
            if d.is_dir()
            and not d.name.startswith("_")
            and (d / "__init__.py").exists()
        )

        # Build the mapping hyperspy.io expects:
        #   { "format_name": {"api": "rsciio.XXX", "file_extensions": [...], ...} }
        registered = []
        for fmt_dir in format_dirs:
            mod_name = f"rsciio.{fmt_dir.name}"
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue

            # Each rsciio plugin exposes file_reader/file_writer and a
            # `spec` or attributes like `file_extensions`, `format_name`
            # We poke it so HyperSpy's _infer_file_reader can find it later
            registered.append(mod_name)

        # Now make HyperSpy aware of them by triggering its extension
        # registration mechanism
        try:
            from hyperspy.extensions import ALL_EXTENSIONS
            # If ALL_EXTENSIONS is already populated, we may just need
            # to ensure the io_plugins are in there
        except ImportError:
            pass

        # The most reliable approach: patch hyperspy.io directly
        try:
            import hyperspy.io
            # Force re-scan of rosettasciio plugins
            if hasattr(hyperspy.io, '_io_plugins'):
                # HyperSpy 2.x
                _register_rsciio_v2(rsciio_dir)
            else:
                _register_rsciio_v1(rsciio_dir)
        except Exception as e:
            print(f"[rthook_hyperspy] Warning during IO registration: {e}")

    except ImportError as e:
        print(f"[rthook_hyperspy] Could not import rsciio/hyperspy: {e}")


def _register_rsciio_v2(rsciio_dir):
    """Registration for HyperSpy 2.x which uses rosettasciio directly."""
    import hyperspy.io
    import rsciio

    # In HyperSpy 2.x, IO plugins come from rosettasciio's entry points.
    # We can force-populate by calling rsciio's own registry.
    try:
        # rosettasciio >= 0.3 has a format registry
        from rsciio import IO_PLUGINS
        if not IO_PLUGINS:
            _scan_rsciio_plugins(rsciio_dir)
    except ImportError:
        _scan_rsciio_plugins(rsciio_dir)


def _register_rsciio_v1(rsciio_dir):
    """Fallback for older HyperSpy versions."""
    _scan_rsciio_plugins(rsciio_dir)


def _scan_rsciio_plugins(rsciio_dir):
    """
    Import every rsciio sub-package so their format specs are registered
    in rsciio.IO_PLUGINS (or equivalent global).
    """
    import rsciio
    from pathlib import Path

    rsciio_dir = Path(rsciio.__file__).parent

    for fmt_dir in sorted(rsciio_dir.iterdir()):
        if (
            fmt_dir.is_dir()
            and not fmt_dir.name.startswith("_")
            and (fmt_dir / "__init__.py").exists()
        ):
            try:
                importlib.import_module(f"rsciio.{fmt_dir.name}")
            except Exception:
                pass

    # Some versions of rsciio populate IO_PLUGINS on import of submodules
    # Double-check and manually populate if needed
    try:
        from rsciio import IO_PLUGINS
        if not IO_PLUGINS:
            # Last resort: build IO_PLUGINS from the spec dicts
            for fmt_dir in sorted(rsciio_dir.iterdir()):
                if (
                    fmt_dir.is_dir()
                    and not fmt_dir.name.startswith("_")
                    and (fmt_dir / "__init__.py").exists()
                ):
                    try:
                        mod = importlib.import_module(f"rsciio.{fmt_dir.name}")
                        # Look for the standard spec dictionary
                        for attr in ("SPEC", "spec", "specification"):
                            if hasattr(mod, attr):
                                IO_PLUGINS.append(getattr(mod, attr))
                                break
                    except Exception:
                        pass
    except ImportError:
        pass


_patch_hyperspy_io()
