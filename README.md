## Status Log
One entry per week (dated by the Monday). Weeks with nothing worth noting are skipped.

### 2026-08-24
- Analysis tab to measure distances and angles directly on the image.
- New gold/cobalt Mask-RCNN weights published to Hugging Face.
- Improve macOS build pipelines. 
- Unified app logging.

### 2026-08-17
- Base Mask-RCNN replaced by YoloMaskRCNN (YOLO detection + mask head), with YOLO running on the selected device.
- Batch CLI gains an extension filter and verbose help.

### 2026-08-10
- Batch CLI: segment a folder of images in one run and get a summary CSV, with include/exclude export presets.
- Synthetic Mask-RCNN model added for testing.

### 2026-07-27
- Histogram export to PNG/SVG/CSV with size stats and fit parameters. 
- Fill-opacity slider available outside refine mode.
- Friendlier errors for failed uploads and malformed TIFFs.

### 2026-07-20
- Shape-label config fully editable from the stats panel.
- Polygon vertex count adapts to particle size.
- Windows build scripts. 

### 2026-07-13
- Freehand scribble tool to mark background pixels for the RF recovery pass, replacing the distance guess.
- Shape labels moved into a user-editable config file.
- Sidebar rebuilt into tabs.
- Enable on demand lazy loading models.  
- Refine-mode hover tooltip with user-picked metrics.

### 2026-06-29
- COCO JSON export, with CVAT import instructions bundled in the zip.
- Git branch recorded in build output for traceability.

### 2026-06-22
- Random-forest recovery pass after YOLO+SAM: a pixel classifier finds missed regions and offers them as accept/discard proposals.
- RF masks used directly as proposals, SAM dropped from the recover step.

### 2026-06-01
- Enter pixel size/units manually for formats without metadata.
- Draw over the image's scale bar to compute pixel size.
- CUDA-enabled Windows build spec.

### 2026-05-25
- Click-to-add: click a missed particle and accept or discard the proposal.
- "Find Similar Particles" matches new regions against already-detected ones.
- Manual point and box annotation canvases for when the model finds nothing.
- Windows build support.

### 2026-05-18
- Refine mode upgrades: rotate polygons with an on-canvas handle, copy-paste polygons, global opacity control.

### 2026-05-11
- SAM box predictions batched to stop RAM crashes on particle-dense images.
- Warning on high particle counts.
- Mask overlays aligned correctly in the desktop build.
- Reduce app bundles size. 

### 2026-04-13
- Stats exportable to CSV.
- Clicking a shape in the stats table highlights every particle of that shape in the workspace.
- Scatter plot polish: jitter, click-legend filtering.

### 2026-04-06
- Stats recomputed from the labelled mask instead of contour approximations.
- Morphological cleanup filters out tiny specks.
- Scatter plot of circularity vs aspect ratio replaces the pie chart.
- Saving refine edits refreshes stats.

### 2026-03-30
- EMD file support with metadata extraction 
- Detailed stats dashboard: per-particle table, histogram, click-to-highlight, and normal/lognormal/Weibull fits with a KS test.

### 2026-03-23
- macOS .app builds via PyInstaller.
- Loading window while model weights boot.
- Weights hosted on Hugging Face.

### 2026-03-16
- Export selected session results to a zip from the sidebar.
- Project ported from pip to uv.
- Desktop app pivots to pywebview.

### 2026-03-09
- Split-instances editing.
- YOLO exported to ONNX with a shared SAM predictor per session.
- Models load once and use GPU when available.
- Weights tracked via git LFS.

### 2026-03-02
- Refine mode: segmentation blobs served as editable polygons you can reshape and save.
- Colour-coded masks.
- Zoom and TIFF support.
- Latency tests confirm SAM is the bottleneck.

### 2026-02-23
- Ground-truth upload with segmentation-vs-truth scoring.
- Black-out regions honoured in both segmentation and scoring.
- .npy image support.
- Dual black-out: keep only, or discard, the selected regions.

### 2026-02-16
- First working version: FastAPI server, Next.js workspace, per-image sessions, and the YoloSAM pipeline (YOLO detection + SAM masks).
- Custom Mask-RCNN model added alongside, with the client fetching the list of available models from the server.
