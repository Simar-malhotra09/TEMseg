export default function Docs() {
  return (
    <div style={{
      maxWidth: 720,
      margin: "0 auto",
      padding: "64px 32px",
      fontFamily: "'Georgia', serif",
      color: "#1a1a1a",
      lineHeight: 1.7,
      fontSize: 16,
      background: "#fff",
      minHeight: "100vh",
    }}>

      {/* ── Overview ── */}
      <section style={sectionStyle}>
        <h2 style={h2Style}>Overview</h2>
        <p>
          TEMseg is a research tool for instance segmentation of nanoparticles in transmission
          electron microscopy (TEM) and scanning TEM (STEM) images. It supports multiple
          segmentation models, interactive mask refinement, region exclusion, and quantitative
          evaluation against ground truth annotations.
        </p>
      </section>

      {/* ── Models ── */}
      <section style={sectionStyle}>
        <h2 style={h2Style}>Models</h2>

        <h3 style={h3Style}>YoloSAM</h3>
        <p>
          A hybrid pipeline combining YOLOv8 object detection with the Segment Anything Model
          (SAM) for precise instance mask generation. YOLO identifies particle bounding boxes;
          SAM generates high-fidelity masks within each box. Model weights were obtained from
          work on automated nanoparticle analysis in STEM imaging —
          see <a href="https://github.com/ArdaGen/STEM-Automated-Nanoparticle-Analysis-YOLOv8-SAM" style={linkStyle}>
            ArdaGen/STEM-Automated-Nanoparticle-Analysis
          </a> and the associated <a href="https://www.sciencedirect.com/science/article/abs/pii/S0304399125000154" style={linkStyle}>
            publication
          </a>.
        </p>
        <p style={noteStyle}>
          Limitation: SAM inference is slow on CPU (~4s per image). GPU acceleration may reduce
          this to ~0.3s. The hybrid approach also means missed YOLO detections propagate —
          particles outside YOLO's confidence threshold will not be segmented regardless of
          SAM's capability.
        </p>

        <h3 style={h3Style}>Mask R-CNN</h3>
        <p>
          A custom Mask R-CNN model trained on TEM particle data. Provides end-to-end instance
          segmentation without a two-stage pipeline. Weights are custom-trained and not publicly
          distributed.
        </p>
        <p style={noteStyle}>
          Limitation: Performance is sensitive to the distribution of training data. Images with
          significantly different contrast, scale, or particle morphology from the training set
          may yield poor results.
        </p>
      </section>

      {/* ── Segmentation ── */}
      <section style={sectionStyle}>
        <h2 style={h2Style}>Segmentation</h2>
        <p>
          Upload an image (TIF, TIFF, NPY, JPEG, PNG), select a model, and click Run
          Segmentation. The resulting mask is overlaid on the original image using a colormap
          for visibility. Segmentation statistics are computed automatically and shown in the
          right panel.
        </p>
        <p>
          Statistics reported include particle count, average particle size (px²), average
          circularity (0–1, where 1 is a perfect circle), and coverage (fraction of image area
          occupied by particles).
        </p>
        <p style={noteStyle}>
          Note: the segmentation mask is a combined binary mask — individual particles are not
          instance-separated at this stage. Use the Refine Masks tool to obtain per-instance
          polygons.
        </p>
      </section>

      {/* ── Region Controls ── */}
      <section style={sectionStyle}>
        <h2 style={h2Style}>Region Controls</h2>

        <h3 style={h3Style}>Exclude Mode</h3>
        <p>
          Draw rectangles over regions to black out before passing the image to the model.
          This is useful for removing scale bars, metadata overlays, or image artifacts that
          were not present in model training data and may confuse detection.
        </p>

        <h3 style={h3Style}>Isolate Mode</h3>
        <p>
          Draw rectangles over regions of interest. Only these regions are passed to the model —
          each as an independent patch. Results are stitched back into a full-size mask. This
          uses YOLO's native batch inference across patches for efficiency, followed by sequential
          SAM inference per patch.
        </p>
        <p style={noteStyle}>
          Limitation: Patch-based inference increases total latency linearly with the number of
          patches (~3–4s per patch on CPU). Particles at patch boundaries may be clipped. Isolate
          mode is currently only supported for YoloSAM.
        </p>
        <p style={noteStyle}>
          Note: region selections are tied to the segmentation result. If regions are modified
          after running segmentation, a warning is shown and the mask should be recomputed for
          results to remain consistent — particularly important when computing ground truth scores.
        </p>
      </section>

      {/* ── Mask Refinement ── */}
      <section style={sectionStyle}>
        <h2 style={h2Style}>Mask Refinement</h2>
        <p>
          After segmentation, click Refine Masks to enter refinement mode. The combined binary
          mask is decomposed into individual instances using connected components analysis. Each
          instance is represented as a simplified polygon (via <code style={codeStyle}>cv2.approxPolyDP</code>)
          and displayed as an editable overlay.
        </p>
        <p>
          Click a polygon to select it and reveal its vertices. Drag individual vertices to
          adjust the boundary. Press Backspace or Delete to remove a false-positive instance.
          Click Save Refinements to rasterize the updated polygons back to a mask.
        </p>
        <p style={noteStyle}>
          Limitation: Connected components treats touching particles as a single instance.
          Particles that are fused in the binary mask will appear as one polygon. Manual vertex
          editing can approximate a split but a dedicated split tool is not yet implemented.
          The polygon simplification epsilon (1% of arc length) balances editability against
          boundary accuracy — very irregular particles may lose fine detail.
        </p>
      </section>

      {/* ── Ground Truth ── */}
      <section style={sectionStyle}>
        <h2 style={h2Style}>Ground Truth Evaluation</h2>
        <p>
          Upload a ground truth mask (NPY, PNG, TIFF) at any time during a
          session. Supported formats are normalized to a binary mask internally. If segmentation
          has already been run, scores are computed immediately. Otherwise, scores are computed
          automatically when segmentation completes.
        </p>
        <p>Metrics reported:</p>
        <ul style={{ paddingLeft: 24, marginTop: 8 }}>
          <li><strong>IoU</strong> — Intersection over Union between predicted and ground truth masks.</li>
          <li><strong>Dice</strong> — Dice coefficient (F1 score on pixel level).</li>
          <li><strong>Pixel Accuracy</strong> — Fraction of pixels correctly classified.</li>
        </ul>
        <p style={noteStyle}>
          Note: scores are computed against the same blackout/isolate regions that were active
          during segmentation, ensuring consistency. A warning is shown if regions are modified
          after the last segmentation run.
        </p>
        <p style={noteStyle}>
          Limitation: evaluation is mask-level, not instance-level. Two masks with the same
          overall coverage but different instance boundaries will yield the same IoU. Instance-level
          matching (e.g. Hungarian matching by IoU threshold) is not yet implemented.
        </p>
      </section>

      {/* ── File Formats ── */}
      <section style={sectionStyle}>
        <h2 style={h2Style}>Supported File Formats</h2>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #ddd" }}>
              <th style={thStyle}>Format</th>
              <th style={thStyle}>Input Image</th>
              <th style={thStyle}>Ground Truth</th>
              <th style={thStyle}>Notes</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["JPEG / PNG", "✓", "✓", "Standard; PNG preferred for masks"],
              ["TIF / TIFF", "✓", "✓", "Multi-channel and Z-stacks supported; first slice used"],
              ["NPY", "✓", "✓", "NumPy binary; normalized to uint8 for display"],
              ["COCO JSON", "—", "Planned", "Polygon annotations not yet supported"],
            ].map(([fmt, img, gt, note]) => (
              <tr key={fmt as string} style={{ borderBottom: "1px solid #f0f0f0" }}>
                <td style={tdStyle}><code style={codeStyle}>{fmt}</code></td>
                <td style={{ ...tdStyle, textAlign: "center" }}>{img}</td>
                <td style={{ ...tdStyle, textAlign: "center" }}>{gt}</td>
                <td style={{ ...tdStyle, color: "#666" }}>{note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* ── Known Limitations ── */}
      <section style={sectionStyle}>
        <h2 style={h2Style}>Known Limitations</h2>
        <ul style={{ paddingLeft: 24 }}>
          <li style={{ marginBottom: 8 }}>
            Inference is slow on CPU (on average ~4s). A GPU impl is strongly recommended for interactive use.
          </li>
          <li style={{ marginBottom: 8 }}>
            Sessions are stored on the local filesystem and cleaned up after 24 hours.
            Reload the page to start a new session.
          </li>
          <li style={{ marginBottom: 8 }}>
            No multi-image sessions; each upload creates a new session.
          </li>
          <li style={{ marginBottom: 8 }}>
            Export functionality is not yet implemented.
          </li>
          <li style={{ marginBottom: 8 }}>
            COCO JSON ground truth format is not yet supported.
          </li>
        </ul>
      </section>

      <p style={{ color: "#aaa", fontSize: 12, marginTop: 64, fontFamily: "monospace" }}>
        last updated — {new Date().toLocaleDateString()}
      </p>
    </div>
  );
}
const sectionStyle: React.CSSProperties = {
  marginBottom: 48,
};

const h2Style: React.CSSProperties = {
  fontSize: 20,
  fontWeight: 700,
  marginBottom: 12,
  marginTop: 0,
  paddingBottom: 6,
  borderBottom: "1px solid #e8e8e8",
};

const h3Style: React.CSSProperties = {
  fontSize: 16,
  fontWeight: 700,
  marginBottom: 6,
  marginTop: 24,
};

const noteStyle: React.CSSProperties = {
  fontSize: 14,
  color: "#666",
  borderLeft: "3px solid #e8e8e8",
  paddingLeft: 12,
  marginTop: 8,
};

const linkStyle: React.CSSProperties = {
  color: "#1a1a1a",
  textDecorationColor: "#aaa",
};

const codeStyle: React.CSSProperties = {
  fontFamily: "monospace",
  fontSize: 13,
  background: "#f5f5f5",
  padding: "1px 4px",
  borderRadius: 3,
};

const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "8px 12px",
  fontWeight: 600,
  fontSize: 13,
};

const tdStyle: React.CSSProperties = {
  padding: "8px 12px",
  verticalAlign: "top",
};
