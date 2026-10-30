import React, { useEffect, useMemo, useState } from "react";

import {

  ShieldCheck,

  UploadCloud,

  Image as ImageIcon,

  Film,

  ScanFace,

  AlertTriangle,

  CheckCircle2,

  LoaderCircle,

  FileJson,

  Trash2,

  ChevronDown,

  Info,

  Sparkles,

  Clock3,

  BarChart3,
  Mail,
  ArrowLeft,
  Code2,

} from "lucide-react";

const API_BASE = (

  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

).replace(/\/+$/, "");

const MAX_IMAGE_BYTES = 12 * 1024 * 1024;

const MAX_VIDEO_BYTES = 80 * 1024 * 1024;

// ============================================================

// Utility functions

// ============================================================

function formatPercent(value) {

  const number = Number(value);

  return Number.isFinite(number)

    ? `${(number * 100).toFixed(1)}%`

    : "—";

}

function formatSeconds(value) {

  const number = Number(value);

  if (!Number.isFinite(number)) return "—";

  return `${number.toFixed(2)} sec`;

}

function formatFileSize(bytes) {

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;

}

function humanize(value) {

  if (!value) return "Not available";

  const labels = {

    fake: "Model suggests AI-generated / fake",
    real: "Model suggests real",
    model_suggests_ai_generated: "Model suggests AI-generated",

    model_suggests_human_or_real: "Model suggests human/real",

    model_predicts_deepfake: "Model suggests an AI-generated / fake-looking face crop (not proof of a face swap)",

    model_predicts_realism: "Model suggests a real-looking face crop (not proof of authenticity)",

    unmapped_label: "Unmapped model label",

    unknown: "Unknown",

    inconclusive: "Inconclusive",

  };

  return (

    labels[value] ||

    String(value)

      .replaceAll("_", " ")

      .replaceAll("-", " ")

  );

}

function getScoreWidth(value) {

  const number = Number(value);

  if (!Number.isFinite(number)) return "0%";

  return `${Math.max(0, Math.min(100, number * 100))}%`;

}

function getFaceCount(result) {

  if (Number.isFinite(result?.face_count)) {

    return result.face_count;

  }

  return (result?.face_regions || []).filter((face) => face.box).length;

}

function downloadJson(result) {

  const blob = new Blob(

    [JSON.stringify(makeDisplayReport(result), null, 2)],

    { type: "application/json" }

  );

  const url = URL.createObjectURL(blob);

  const anchor = document.createElement("a");

  anchor.href = url;

  anchor.download = "media-forensics-report.json";

  document.body.appendChild(anchor);

  anchor.click();

  anchor.remove();

  window.setTimeout(() => URL.revokeObjectURL(url), 1000);

}

function printReport() {
  // Opens the browser print dialog; choose "Save as PDF" as the destination.
  window.print();
}

function getClassScore(classification, acceptedLabels) {
  const predictions = Array.isArray(classification?.predictions)
    ? classification.predictions
    : [];
  const accepted = new Set(acceptedLabels.map((label) => label.toLowerCase()));
  const match = predictions.find((item) =>
    accepted.has(String(item.label || "").trim().toLowerCase().replaceAll("_", " ").replaceAll("-", " "))
  );
  const score = Number(match?.score);
  return Number.isFinite(score) ? score : null;
}

function getFrameSignals(frame) {
  const artificial = getClassScore(frame?.classification, [
    "fake", "artificial", "ai generated", "synthetic", "generated",
  ]);
  const faces = (Array.isArray(frame?.face_results) ? frame.face_results : [])
    .map((face) => ({
      trackId: face.track_id,
      score: getClassScore(face.classification, ["fake", "deepfake", "deep fake"]),
      interpretation: face.interpretation,
    }))
    .filter((face) => face.score !== null);
  return { artificial, faces };
}

// ============================================================

// Prediction score bars

// ============================================================

// Hide face-model provider identifiers from the frontend technical report.

// The API response and analysis itself remain unchanged.

function makeDisplayReport(result) {

  const copy = JSON.parse(JSON.stringify(result));

  if (copy.analysis_method) delete copy.analysis_method.face_model_id;

  const hideFaceModelId = (face) => {

    if (face?.classification) delete face.classification.model_id;

  };

  (copy.face_regions || []).forEach(hideFaceModelId);

  (copy.frame_results || []).forEach((frame) => {

    (frame.face_results || []).forEach(hideFaceModelId);

  });

  return copy;

}

function ScoreList({ predictions }) {

  if (!Array.isArray(predictions) || predictions.length === 0) {

    return <p className="muted">No prediction scores available.</p>;

  }

  return (

    <div className="score-list">

      {predictions.slice(0, 5).map((prediction, index) => (

        <div

          className="score-row"

          key={`${prediction.label || "label"}-${index}`}

        >

          <div className="score-label">

            <span>{prediction.label || "Unknown"}</span>

            <strong>{formatPercent(prediction.score)}</strong>

          </div>

          <div className="track">

            <div

              className="fill"

              style={{ width: getScoreWidth(prediction.score) }}

            />

          </div>

        </div>

      ))}

    </div>

  );

}

// ============================================================

// Reusable model summary

// ============================================================

function ModelSummary({ classification, title, subtitle }) {

  if (!classification) return null;

  return (

    <div className="model-summary">

      <div className="section-heading">

        <div>

          <h4>{title}</h4>

          {subtitle && <small>{subtitle}</small>}

        </div>

        <span className="pill">

          {formatPercent(classification.top_score)}

        </span>

      </div>

      <p>

        Top prediction:{" "}

        <strong>{classification.top_label || "Unknown"}</strong>

      </p>

      <ScoreList predictions={classification.predictions} />


    </div>

  );

}

// ============================================================

// Face region result card

// ============================================================

function FaceRegionCard({ face }) {

  const box = face.box;

  const classification = face.classification;

  return (

    <article className="region-card">

      <div className="region-card-heading">

        <div>

          <strong>{humanize(face.region)}</strong>

          {box && (

            <small>

              Position: {box.x}, {box.y} · Size: {box.width} × {box.height}

            </small>

          )}

        </div>

        {classification ? (

          <span className="pill">

            {formatPercent(classification.top_score)}

          </span>

        ) : (

          <span className="pill warning">Unavailable</span>

        )}

      </div>

      {classification ? (

        <>

          <div className="face-prediction">

            <ScanFace size={20} />

            <div>

              <strong>{classification.top_label || "Unknown"}</strong>

              <small>

                Model interpretation: {humanize(face.interpretation)}

              </small>

            </div>

          </div>

          <ScoreList predictions={classification.predictions} />

        </>

      ) : (

        <div className="error-message">

          <AlertTriangle size={16} />

          <span>{face.error || "No face prediction was returned."}</span>

        </div>

      )}

      {face.warning && (

        <p className="disclaimer">

          <Info size={15} />

          {face.warning}

        </p>

      )}

    </article>

  );

}

// ============================================================

// Video frame result

// ============================================================

function VideoFrameCard({ frame }) {

  const classification = frame.classification;

  return (

    <article className="frame-row" id={`frame-${frame.frame_index}`} key={frame.frame_index}>

      <div className="frame-time">

        <span>Frame #{frame.frame_index}</span>

        <small>{formatSeconds(frame.timestamp_seconds)}</small>

      </div>

      {classification ? (

        <>

          <div className="frame-label">

            <strong>{classification.top_label || "Unknown"}</strong>

            <small>{formatPercent(classification.top_score)}</small>

          </div>

          <div className="mini-track">

            <div

              className="fill"

              style={{

                width: getScoreWidth(classification.top_score),

              }}

            />

          </div>

          <small className="muted">

            {humanize(frame.interpretation)}

          </small>

        </>

      ) : (

        <div className="error-message">

          <AlertTriangle size={16} />

          <span>{frame.error || "Frame analysis unavailable."}</span>

        </div>

      )}

      {Array.isArray(frame.face_results) && frame.face_results.some((face) => face.track_id) && (

        <div className="muted">

          {frame.face_results.filter((face) => face.track_id).map((face, index) => (

            <span key={`${face.track_id}-${index}`}>

              {index > 0 ? " · " : ""}

              {face.track_id}: {humanize(face.interpretation)}

              {face.classification ? ` (${formatPercent(face.classification.top_score)})` : ""}

            </span>

          ))}

        </div>

      )}

    </article>

  );

}

// ============================================================

// Video summary

// ============================================================

// Video summaries: face predictions and heuristic temporal tracking.

function FaceAnalysisSummary({ summary }) {

  if (!summary) return null;

  const stats = [

    ["Frames with faces", summary.frames_with_detected_faces],

    ["Detected face regions", summary.detected_face_regions],

    ["Classified face regions", summary.successfully_classified_face_regions],

    ["Failed face classifications", summary.failed_face_classifications],

  ];

  return (

    <div className="model-summary">

      <div className="section-heading">

        <div>

          <h4>Face-crop analysis summary</h4>

          <small>Fake-vs-real estimates from detected face crops</small>

        </div>

        <span className="pill warning">

          {humanize(summary.face_signal_verdict || "inconclusive")}

        </span>

      </div>

      <div className="stats-grid">

        {stats.map(([label, value]) => (

          <div className="stat" key={label}>

            <span>{label}</span>

            <strong>{value ?? "—"}</strong>

          </div>

        ))}

      </div>

      <p>

        Mean face-crop fake-class score:{" "}

        <strong>{formatPercent(summary.mean_face_deepfake_score)}</strong>

      </p>

      <p>

        Mean face-crop real-class score:{" "}

        <strong>{formatPercent(summary.mean_face_realism_score)}</strong>

      </p>

      {summary.face_signal_reason && <p className="muted">{summary.face_signal_reason}</p>}

      {summary.note && <p className="disclaimer"><Info size={15} /> {summary.note}</p>}

    </div>

  );

}

function TemporalAnalysisSummary({ summary }) {

  if (!summary) return null;

  const tracks = Array.isArray(summary.tracks) ? summary.tracks : [];

  return (

    <div className="model-summary">

      <div className="section-heading">

        <div>

          <h4>Temporal face tracking</h4>

          <small>Heuristic association across sampled frames</small>

        </div>

        <span className="pill">{summary.tracks_found ?? tracks.length} tracks</span>

      </div>

      <div className="stats-grid">

        <div className="stat">

          <span>Tracks found</span>

          <strong>{summary.tracks_found ?? "—"}</strong>

        </div>

        <div className="stat">

          <span>Seen in multiple frames</span>

          <strong>{summary.tracks_seen_in_multiple_sampled_frames ?? "—"}</strong>

        </div>

        <div className="stat">

          <span>High score variability</span>

          <strong>{summary.tracks_with_high_face_score_variability ?? "—"}</strong>

        </div>

      </div>

      {tracks.length ? (

        <div className="face-results-list">

          {tracks.map((track, index) => (

            <article className="region-card" key={track.track_id || index}>

              <div className="region-card-heading">

                <div>

                  <strong>{track.track_id || `Face track ${index + 1}`}</strong>

                  <small>

                    {track.observations ?? 0} observation(s)

                    {Array.isArray(track.timestamps_seconds) && track.timestamps_seconds.length

                      ? ` · ${track.timestamps_seconds.map((time) => `${Number(time).toFixed(2)}s`).join(", ")}`

                      : ""}

                  </small>

                </div>

                <span className={`pill ${track.score_variability_flag ? "warning" : ""}`}>

                  {track.score_variability_flag ? "Variable scores" : "No high variability flag"}

                </span>

              </div>

              <p>

                Mean face-crop fake-class score:{" "}

                <strong>{formatPercent(track.mean_deepfake_score)}</strong>

              </p>

              <p>

                Face-crop fake-score range:{" "}

                <strong>{formatPercent(track.deepfake_score_range)}</strong>

              </p>

            </article>

          ))}

        </div>

      ) : (

        <p className="muted">No face tracks were available for this video.</p>

      )}

      {summary.note && <p className="disclaimer"><Info size={15} /> {summary.note}</p>}

    </div>

  );

}


function EvidenceTimeline({ result }) {
  const frames = Array.isArray(result.frame_results) ? result.frame_results : [];
  const threshold = Number(
    result.evidence_summary?.threshold ??
    result.summary?.verdict_threshold ??
    0.65
  );
  const noteworthy = Array.isArray(result.evidence_summary?.noteworthy_signals)
    ? result.evidence_summary.noteworthy_signals
    : [];

  if (!frames.length) {
    return (
      <div className="model-summary">
        <h4>Evidence timeline</h4>
        <p className="muted">No sampled frame data is available.</p>
      </div>
    );
  }

  return (
    <div className="model-summary evidence-timeline">
      <div className="section-heading">
        <div>
          <h4><Clock3 size={17} /> Evidence timeline</h4>
          <small>Sampled frames and available face-analysis signals</small>
        </div>
      </div>
      <p className="muted">
        Highlighted markers mean a model score crossed the configured heuristic
        threshold. They are review cues, not proof of manipulation.
      </p>
      <div className="timeline-track" role="list" aria-label="Sampled video frame evidence">
        {frames.map((frame, index) => {
          const signals = getFrameSignals(frame);
          const frameElevated = signals.artificial !== null && signals.artificial >= threshold;
          const elevatedFaces = signals.faces.filter((face) => face.score >= threshold);
          const elevated = frameElevated || elevatedFaces.length > 0;
          return (
            <a
              className={`timeline-marker ${elevated ? "elevated" : ""} ${frame.error ? "failed" : ""}`}
              href={`#frame-${frame.frame_index ?? index}`}
              key={`${frame.frame_index ?? "frame"}-${index}`}
              role="listitem"
              title={`Frame ${frame.frame_index ?? index} at ${formatSeconds(frame.timestamp_seconds)}${elevated ? " — elevated signal" : ""}`}
            >
              <span className="timeline-dot" />
              <strong>F{frame.frame_index ?? index}</strong>
              <small>{formatSeconds(frame.timestamp_seconds)}</small>
              {frameElevated && <span className="evidence-chip">Image {formatPercent(signals.artificial)}</span>}
              {elevatedFaces.map((face, faceIndex) => (
                <span className="evidence-chip" key={`${face.trackId || "face"}-${faceIndex}`}>
                  Face {formatPercent(face.score)}
                </span>
              ))}
              {!elevated && !frame.error && <span className="muted timeline-neutral">No elevated signal</span>}
              {frame.error && <span className="muted timeline-neutral">Analysis failed</span>}
            </a>
          );
        })}
      </div>
      {noteworthy.length > 0 && (
        <div className="evidence-list">
          <h4>Flagged evidence items</h4>
          {noteworthy.map((item, index) => (
            <div className="evidence-item" key={`${item.type || "signal"}-${item.frame_index}-${index}`}>
              <div>
                <strong>
                  {item.type === "face_model_signal" ? "Face-model signal" : "Frame-model signal"}
                  {" · "}Frame {item.frame_index}
                </strong>
                <small>{formatSeconds(item.timestamp_seconds)} · Score {formatPercent(item.score)} · Threshold {formatPercent(item.threshold ?? threshold)}</small>
              </div>
              <a href={`#frame-${item.frame_index}`}>View frame</a>
              <p>{item.description || "Elevated model score flagged for review."}</p>
            </div>
          ))}
        </div>
      )}
      <p className="disclaimer">
        <Info size={15} />
        {result.evidence_summary?.interpretation_note ||
          "Scores are model outputs, not calibrated probabilities or proof of authenticity."}
      </p>
    </div>
  );
}

function VideoSummary({ result }) {

  const summary = result.summary || {};

  const info = result.video_info || {};

  const analyzedFrames =

    summary.analyzed_frames ?? info.sampled_frames ?? 0;

  const successfulFrames =

    summary.successful_frames ??

    (result.frame_results || []).filter(

      (frame) => frame.classification

    ).length;

  const failedFrames =

    summary.failed_frames ??

    (result.frame_results || []).filter(

      (frame) => frame.error

    ).length;

  const labelCounts = summary.label_counts || {};

  const verdict = summary.verdict || "inconclusive";
  const verdictReason = summary.verdict_reason || summary.frame_signal_reason ||
    "The backend did not return a specific reason for the assessment.";
  const verdictClass = verdict === "likely_fake" ? "verdict-caution" :
    verdict === "likely_real" ? "verdict-neutral" : "verdict-inconclusive";

  return (

    <>

      <section className={`final-verdict ${verdictClass}`}>
        <div className="verdict-icon"><ShieldCheck size={23} /></div>
        <div className="verdict-copy">
          <p className="eyebrow">HEURISTIC ASSESSMENT</p>
          <h3>{humanize(verdict)}</h3>
          <p>{verdictReason}</p>
          <small>This is an experimental model-based assessment, not proof that a video is authentic or manipulated.</small>
        </div>
      </section>

      <EvidenceTimeline result={result} />

      <div className="stats-grid">

        <div className="stat">

          <span>Frames sampled</span>

          <strong>{analyzedFrames}</strong>

        </div>

        <div className="stat">

          <span>Successful analyses</span>

          <strong>{successfulFrames}</strong>

        </div>

        <div className="stat">

          <span>Failed frames</span>

          <strong>{failedFrames}</strong>

        </div>

        <div className="stat">

          <span>Duration</span>

          <strong>

            {info.duration_seconds != null

              ? `${info.duration_seconds}s`

              : "—"}

          </strong>

        </div>

        <div className="stat">

          <span>Frame rate</span>

          <strong>

            {info.fps != null ? `${info.fps} FPS` : "—"}

          </strong>

        </div>

        <div className="stat">

          <span>Video resolution</span>

          <strong>

            {info.width && info.height

              ? `${info.width} × ${info.height}`

              : "—"}

          </strong>

        </div>

      </div>

      <div className="model-summary">

        <div className="section-heading">

          <div>

            <h4>

              <BarChart3 size={17} /> Frame prediction summary

            </h4>

            <small>Counts of model predictions across sampled frames</small>

          </div>

          <span className="pill warning">

            {humanize(summary.verdict || "inconclusive")}

          </span>

        </div>

        {Object.keys(labelCounts).length > 0 ? (

          <div className="score-list">

            {Object.entries(labelCounts).map(([label, count]) => (

              <div className="score-row" key={label}>

                <div className="score-label">

                  <span>{label}</span>

                  <strong>

                    {count} frame{count === 1 ? "" : "s"}

                  </strong>

                </div>

                <div className="track">

                  <div

                    className="fill"

                    style={{

                      width: getScoreWidth(

                        analyzedFrames > 0 ? count / analyzedFrames : 0

                      ),

                    }}

                  />

                </div>

              </div>

            ))}

          </div>

        ) : (

          <p className="muted">No frame labels were returned.</p>

        )}

        {summary.mean_artificial_class_score != null && (

          <p>

            Mean fake-class score:{" "}

            <strong>

              {formatPercent(summary.mean_artificial_class_score)}

            </strong>

          </p>

        )}

        {summary.mean_human_class_score != null && (

          <p>

            Mean human-class score:{" "}

            <strong>

              {formatPercent(summary.mean_human_class_score)}

            </strong>

          </p>

        )}

        {summary.verdict_reason && (

          <p className="muted">{summary.verdict_reason}</p>

        )}

        {summary.final_verdict_basis && (

          <p className="muted">Verdict basis: {summary.final_verdict_basis}</p>

        )}

        {summary.verdict_threshold != null && (

          <p className="muted">Heuristic threshold: {formatPercent(summary.verdict_threshold)}</p>

        )}

      </div>

      <FaceAnalysisSummary summary={result.face_analysis_summary} />

      <TemporalAnalysisSummary summary={result.temporal_analysis_summary} />

      <h3 className="subheading">Sampled frame results</h3>

      <div className="frame-list">

        {(result.frame_results || []).map((frame, index) => (

          <VideoFrameCard

            key={`${frame.frame_index ?? "frame"}-${index}`}

            frame={frame}

          />

        ))}

      </div>

      <div className="notice-card">

        <Info size={19} />

        <p>

          <strong>Frame analysis is not a final video verdict.</strong>

          <br />

          {result.warning ||

            "Only sampled frames were analyzed. Temporal video manipulation was not checked."}

        </p>

      </div>

    </>

  );

}

// ============================================================

// Results panel

// ============================================================

function ResultPanel({ result, onClear, mediaPreview, videoThumbnail }) {

  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {

    setShowRaw(false);

  }, [result]);

  if (!result) return null;

  const isVideo = result.media_type === "video";

  const overall = result.overall;

  const faceRegions = result.face_regions || [];

  const faceCount = getFaceCount(result);

  const warnings = Array.isArray(result.warnings)

    ? result.warnings

    : [];

  const limitations = Array.isArray(result.limitations)

    ? result.limitations

    : [];

  return (

    <section className="results panel">

      <div className="section-heading">

        <div>

          <p className="eyebrow">ANALYSIS REPORT</p>

          <h2>Results</h2>

        </div>

        <div className="action-row">

          <button

            type="button"

            className="button secondary small"

            onClick={() => downloadJson(result)}

          >

            <FileJson size={16} />

            Export JSON

          </button>

          <button
            type="button"
            className="button secondary small no-print"
            onClick={printReport}
            title="Open print dialog and choose Save as PDF"
          >
            Download PDF
          </button>

          <button

            type="button"

            className="icon-button"

            title="Clear results"

            aria-label="Clear results"

            onClick={onClear}

          >

            <Trash2 size={17} />

          </button>

        </div>

      </div>

      <div className="result-file">

        {isVideo ? <Film size={20} /> : <ImageIcon size={20} />}

        <div>

          <strong>{result.filename || "Uploaded media"}</strong>

          <small>

            {isVideo

              ? "Video frame sampling"

              : `${result.dimensions?.width || "?"} × ${

                  result.dimensions?.height || "?"

                } image`}

          </small>

        </div>

      </div>

      {(isVideo ? videoThumbnail : mediaPreview) && (
        <div className="report-media">
          <h4>Uploaded {isVideo ? "video thumbnail" : "image"}</h4>
          <img
            src={isVideo ? videoThumbnail : mediaPreview}
            alt={isVideo ? "Thumbnail captured from the uploaded video" : "Uploaded image included in the report"}
          />
          <small>{isVideo ? "Representative frame captured from the uploaded video." : "Original uploaded image."}</small>
        </div>
      )}

      {result.report_metadata && (
        <div className="model-summary report-metadata">
          <div className="section-heading">
            <div>
              <h4>Analysis metadata</h4>
              <small>Report and processing details</small>
            </div>
            <span className="pill">{result.report_metadata.report_version || "Report"}</span>
          </div>
          <div className="metadata-grid">
            <div><span>Generated (UTC)</span><strong>{result.report_metadata.generated_at_utc || "Not provided"}</strong></div>
            <div><span>Media type</span><strong>{result.report_metadata.media_type || result.media_type || "Unknown"}</strong></div>
            <div><span>File</span><strong>{result.report_metadata.filename || result.filename || "Uploaded media"}</strong></div>
          </div>
        </div>
      )}

      {/* Image results */}

      {!isVideo && (

        <>

          <section className="final-verdict verdict-inconclusive">
            <div className="verdict-icon"><ShieldCheck size={23} /></div>
            <div className="verdict-copy">
              <p className="eyebrow">IMAGE ASSESSMENT</p>
              <h3>Separate model signals — no combined verdict</h3>
              <p>{result.assessment_policy?.reason || "The overall image classifier and face-crop classifier are separate signals and are not combined into one authenticity verdict."}</p>
              <small>A model score does not prove that an image is genuine or manipulated.</small>
            </div>
          </section>

          {overall && (

            <>

              <div className="result-summary">

                <div className="summary-icon">

                  <Sparkles size={22} />

                </div>

                <div className="summary-copy">

                  <span className="muted">Overall image model</span>

                  <h3>{overall.top_label || "Unknown"}</h3>

                  <p>

                    Model score:{" "}

                    <strong>{formatPercent(overall.top_score)}</strong>

                  </p>

                  <small>

                    Interpretation: {humanize(result.interpretation)}

                  </small>

                </div>

              </div>

              <h3 className="subheading">Overall image predictions</h3>

              <ModelSummary

                classification={overall}

                title="AI-image classification"

                subtitle="Analysis of the complete image"

              />

            </>

          )}

          <div className="region-section">

            <div className="section-heading">

              <div>

                <h3 className="subheading">

                  <ScanFace size={18} />

                  Face-crop AI/fake-vs-real signal

                </h3>

                <p className="muted">

                  {faceCount} face{faceCount === 1 ? "" : "s"} detected

                </p>

              </div>

            </div>

            {faceRegions.length > 0 ? (

              <div className="face-results-list">

                {faceRegions.map((face, index) => (

                  <FaceRegionCard

                    key={`${face.region || "face"}-${index}`}

                    face={face}

                  />

                ))}

              </div>

            ) : (

              <p className="muted">

                No face regions were returned. The detector may not have

                found a clearly visible face.

              </p>

            )}

            {result.region_analysis_note && (

              <p className="muted">{result.region_analysis_note}</p>

            )}

          </div>

          <div className="notice-card">

            <Info size={19} />

            <p>

              <strong>Interpret results separately</strong>

              <br />

              The overall image classifier and face classifier are

              different models. Their scores are not combined into a

              single verdict. A face-crop "fake" prediction is an
              AI-generated-vs-real estimate, not a dedicated face-swap finding.
              A "real" prediction also does not prove authenticity.

            </p>

          </div>

        </>

      )}

      {/* Video results */}

      {isVideo && <VideoSummary result={result} />}

      {/* Warnings */}

      {warnings.length > 0 && (

        <div className="limitations">

          <h3>

            <Info size={17} />

            Important warnings

          </h3>

          <ul>

            {warnings.map((warning, index) => (

              <li key={index}>{warning}</li>

            ))}

          </ul>

        </div>

      )}

      {/* Limitations */}

      {limitations.length > 0 && (

        <div className="limitations">

          <h3>

            <AlertTriangle size={17} />

            Limitations

          </h3>

          <ul>

            {limitations.map((item, index) => (

              <li key={index}>{item}</li>

            ))}

          </ul>

        </div>

      )}

      <button

        type="button"

        className="raw-toggle"

        onClick={() => setShowRaw((previous) => !previous)}

        aria-expanded={showRaw}

      >

        Technical report

        <ChevronDown

          size={16}

          className={showRaw ? "rotated" : ""}

        />

      </button>

      {showRaw && (

        <pre className="raw-json">

          {JSON.stringify(makeDisplayReport(result), null, 2)}

        </pre>

      )}

    </section>

  );

}

// ============================================================

// Main application

// ============================================================

export default function App() {

  const [file, setFile] = useState(null);

  const [mode, setMode] = useState("image");

  const [preview, setPreview] = useState("");
  const [videoThumbnail, setVideoThumbnail] = useState("");

  const [maxFrames, setMaxFrames] = useState(12);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  const [result, setResult] = useState(null);

  const [dragging, setDragging] = useState(false);
  const [showCredits, setShowCredits] = useState(false);

  useEffect(() => {
    setVideoThumbnail("");
    if (!preview || mode !== "video") return undefined;

    let cancelled = false;
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;

    const captureFrame = () => {
      if (cancelled || !video.videoWidth || !video.videoHeight) return;
      try {
        const canvas = document.createElement("canvas");
        const scale = Math.min(1, 960 / video.videoWidth);
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        const context = canvas.getContext("2d");
        if (!context) return;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (!cancelled) setVideoThumbnail(canvas.toDataURL("image/jpeg", 0.85));
      } catch {
        // Some browser codecs do not allow thumbnail extraction.
      }
    };

    video.addEventListener("loadeddata", captureFrame, { once: true });
    video.addEventListener("seeked", captureFrame, { once: true });
    video.src = preview;
    video.load();

    return () => {
      cancelled = true;
      video.removeEventListener("loadeddata", captureFrame);
      video.removeEventListener("seeked", captureFrame);
      video.pause();
      video.removeAttribute("src");
      video.load();
    };
  }, [preview, mode]);

  const accepted = useMemo(

    () =>

      mode === "image"

        ? "image/png,image/jpeg,image/webp"

        : "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-m4v",

    [mode]

  );

  const fileLabel = useMemo(

    () =>

      mode === "image"

        ? "PNG, JPG or WebP · up to 12 MB"

        : "MP4, MOV, AVI or WebM · up to 80 MB",

    [mode]

  );

  function releasePreview() {

    if (preview) {

      URL.revokeObjectURL(preview);

    }

  }

  function clearAll() {

    releasePreview();

    setFile(null);

    setPreview("");
    setVideoThumbnail("");

    setResult(null);

    setError("");

  }

  function setSelectedFile(nextFile) {

    setError("");

    setResult(null);

    if (!nextFile) return;

    const isImage = nextFile.type.startsWith("image/");

    const isVideo = nextFile.type.startsWith("video/");

    if (mode === "image" && !isImage) {

      setError("Please choose a JPG, PNG, or WebP image.");

      return;

    }

    if (mode === "video" && !isVideo) {

      setError("Please choose a supported video file.");

      return;

    }

    const maxBytes =

      mode === "image" ? MAX_IMAGE_BYTES : MAX_VIDEO_BYTES;

    if (nextFile.size > maxBytes) {

      setError(

        mode === "image"

          ? "Image exceeds the 12 MB limit."

          : "Video exceeds the 80 MB limit."

      );

      return;

    }

    releasePreview();

    setFile(nextFile);

    setPreview(URL.createObjectURL(nextFile));
    setVideoThumbnail("");

  }

  function changeMode(nextMode) {

    releasePreview();

    setMode(nextMode);

    setFile(null);

    setPreview("");
    setVideoThumbnail("");

    setResult(null);

    setError("");

  }

  async function analyze() {

    if (!file) {

      setError("Choose a file first.");

      return;

    }

    setLoading(true);

    setError("");

    setResult(null);

    try {

      const body = new FormData();

      body.append("file", file);

      const endpoint =

        mode === "image"

          ? "/api/analyze/image"

          : "/api/analyze/video";

      if (mode === "video") {

        body.append("max_frames", String(maxFrames));

      }

      const response = await fetch(`${API_BASE}${endpoint}`, {

        method: "POST",

        body,

      });

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {

        const detail =

          typeof payload.detail === "string"

            ? payload.detail

            : `Request failed (${response.status}). Check the backend terminal.`;

        throw new Error(detail);

      }

      setResult(payload);

    } catch (exception) {

      setError(

        exception.message ||

          "Analysis failed. Check that the backend is running and the API URL is correct."

      );

    } finally {

      setLoading(false);

    }

  }

  function handleDrop(event) {

    event.preventDefault();

    setDragging(false);

    const dropped = event.dataTransfer.files?.[0];

    if (dropped) {

      setSelectedFile(dropped);

    }

  }

  return (

    <div className="app-shell">
      <style>{`
        .final-verdict { display:flex; gap:16px; align-items:flex-start; padding:20px; margin:18px 0; border:1px solid var(--border, #d7dce2); border-radius:16px; background:var(--panel, #fff); }
        .final-verdict .verdict-icon { display:flex; align-items:center; justify-content:center; width:44px; height:44px; border-radius:12px; background:rgba(100,116,139,.12); flex-shrink:0; }
        .final-verdict h3 { margin:3px 0 8px; text-transform:capitalize; }
        .final-verdict p { margin:5px 0 9px; }
        .final-verdict small { display:block; line-height:1.5; opacity:.8; }
        .verdict-caution { border-color:#d97706; background:rgba(217,119,6,.06); }
        .verdict-neutral { border-color:#64748b; }
        .verdict-inconclusive { border-style:dashed; }
        .evidence-timeline { overflow:hidden; }
        .timeline-track { display:flex; gap:10px; overflow-x:auto; padding:14px 2px 18px; align-items:stretch; }
        .timeline-marker { color:inherit; text-decoration:none; min-width:118px; max-width:160px; display:flex; flex-direction:column; gap:6px; padding:12px; border:1px solid var(--border, #d7dce2); border-radius:12px; background:var(--panel, #fff); }
        .timeline-marker.elevated { border:2px solid #d97706; background:rgba(217,119,6,.08); }
        .timeline-marker.failed { border-style:dashed; opacity:.75; }
        .timeline-dot { width:12px; height:12px; border-radius:50%; background:#64748b; display:block; }
        .timeline-marker.elevated .timeline-dot { background:#d97706; box-shadow:0 0 0 4px rgba(217,119,6,.15); }
        .timeline-marker strong { font-size:.9rem; }
        .timeline-marker small, .timeline-neutral { font-size:.76rem; }
        .evidence-chip { display:inline-block; border-radius:6px; padding:3px 6px; font-size:.74rem; background:rgba(217,119,6,.16); }
        .evidence-list { margin-top:14px; }
        .evidence-list h4 { margin-bottom:10px; }
        .evidence-item { padding:12px 0; border-top:1px solid var(--border, #d7dce2); }
        .evidence-item div { display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; }
        .evidence-item small { display:block; opacity:.75; margin-top:4px; }
        .evidence-item p { margin:7px 0 0; }
        .metadata-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }
        .metadata-grid div { display:flex; flex-direction:column; gap:5px; min-width:0; }
        .metadata-grid span { font-size:.8rem; opacity:.72; }
        .metadata-grid strong { overflow-wrap:anywhere; font-size:.9rem; }
        .report-media { margin:16px 0; padding:14px; border:1px solid var(--border, #d7dce2); border-radius:14px; }
        .report-media h4 { margin:0 0 10px; }
        .report-media img { display:block; width:auto; max-width:100%; max-height:420px; object-fit:contain; margin:0 auto; border-radius:8px; }
        .report-media small { display:block; margin-top:8px; text-align:center; opacity:.75; }
        .topbar { gap:16px; flex-wrap:wrap; }
        .topbar-right { display:flex; align-items:center; gap:12px; }
        .brand { text-decoration:none; }
        .brand-accent { color:var(--accent, #6478ff); }
        .nav-button { border:1px solid var(--border, #d7dce2); background:var(--panel, #fff); color:inherit; border-radius:10px; padding:9px 14px; font:inherit; font-weight:600; cursor:pointer; }
        .nav-button:hover { border-color:var(--accent, #6478ff); }
        .credits-page { max-width:900px; margin:34px auto 60px; }
        .credits-card { border:1px solid var(--border, #d7dce2); border-radius:22px; padding:clamp(22px,4vw,42px); background:var(--panel, #fff); box-shadow:0 12px 36px rgba(15,23,42,.06); }
        .credits-profile { text-align:center; padding:12px 0 28px; }
        .credits-avatar { width:76px; height:76px; margin:0 auto 16px; display:grid; place-items:center; border-radius:22px; background:linear-gradient(135deg,#172554,#4f46e5); color:white; }
        .credits-profile h1 { margin:0 0 8px; font-size:clamp(1.8rem,4vw,2.5rem); }
        .credits-role { margin:0; font-weight:650; }
        .credits-email { display:inline-flex; align-items:center; gap:8px; margin-top:14px; color:inherit; text-decoration:none; overflow-wrap:anywhere; }
        .credits-section { border-top:1px solid var(--border, #d7dce2); padding:22px 0 4px; }
        .credits-section h2 { margin:0 0 12px; font-size:1.15rem; }
        .credits-section p { line-height:1.75; }
        .credits-stack { display:flex; flex-wrap:wrap; gap:9px; }
        .credits-stack span { border:1px solid var(--border, #d7dce2); border-radius:999px; padding:7px 12px; font-size:.9rem; }
        .credits-back { display:inline-flex; align-items:center; gap:8px; margin-bottom:16px; }
        @media (max-width:600px) { .topbar-right { width:100%; justify-content:space-between; } }
        @media print {
          .report-media { break-inside:avoid; border-color:#bbb !important; }
          .report-media img { max-height:360px; }
          body { background:#fff !important; color:#111 !important; }
          .topbar, .hero, .workspace, .bottom-strip, footer, .no-print,
          .action-row, .raw-toggle { display:none !important; }
          .main-content { max-width:none !important; padding:0 !important; }
          .results, .panel, .model-summary, .region-card, .final-verdict { box-shadow:none !important; break-inside:avoid; border-color:#bbb !important; }
          .results { display:block !important; border:0 !important; padding:0 !important; }
          .timeline-track { overflow:visible !important; flex-wrap:wrap; }
          .timeline-marker { min-width:90px; }
          .raw-json { white-space:pre-wrap; overflow-wrap:anywhere; }
        }
      `}</style>

      <header className="topbar">

        <a className="brand" href="#">

          <span className="brand-mark">

            <ShieldCheck size={22} />

          </span>

          <span>

            Media<span className="brand-accent">ForensicsAI</span>

          </span>

        </a>

        <div className="topbar-right">
          <span className="status-dot" />
          <span>AI-Powered Media Analysis</span>
          <button
            type="button"
            className="nav-button"
            onClick={() => setShowCredits((current) => !current)}
          >
            {showCredits ? "Home" : "Credits"}
          </button>
        </div>

      </header>

      <main className="main-content">
        {showCredits ? (
          <section className="credits-page">
            <button
              type="button"
              className="nav-button credits-back"
              onClick={() => setShowCredits(false)}
            >
              <ArrowLeft size={16} /> Back to Home
            </button>
            <div className="credits-card">
              <div className="credits-profile">
                <div className="credits-avatar"><Code2 size={34} /></div>
                <h1>Nehan Shaikh</h1>
                <p className="credits-role">Software Engineer (AI &amp; ML)</p>
                <a className="credits-email" href="mailto:nehanshaikh07@gmail.com">
                  <Mail size={17} /> nehanshaikh07@gmail.com
                </a>
                <p className="muted">Developer &amp; Creator of MediaForensicsAI</p>
              </div>
              <div className="credits-section">
                <h2>About MediaForensicsAI</h2>
                <p>
                  MediaForensicsAI is an AI-powered media analysis platform that examines
                  images and videos for model-detected indicators associated with AI
                  generation and possible digital manipulation. It provides separate
                  image, face-level, and sampled-video-frame analysis to support
                  informed review.
                </p>
              </div>
              <div className="credits-section">
                <h2>Technology Stack</h2>
                <div className="credits-stack">
                  {[
                    "React.js", "Python", "FastAPI", "Machine Learning",
                    "Deep Learning", "Computer Vision", "OpenCV",
                    "ONNX Runtime", "Hugging Face Hub", "Render", "Netlify"
                  ].map((technology) => <span key={technology}>{technology}</span>)}
                </div>
              </div>
              <div className="credits-section">
                <h2>Project Credits</h2>
                <p>
                  Concept, development, and integration by <strong>Nehan Shaikh</strong>.
                  The project uses pretrained machine-learning models and open-source
                  software libraries. Model outputs are estimates and should not be
                  treated as definitive proof of authenticity or manipulation.
                </p>
              </div>
            </div>
          </section>
        ) : (
          <>
        <section className="hero">

          <div className="hero-badge">

            <Sparkles size={14} />

            AI-POWERED MEDIA ANALYSIS

          </div>

          <h1>

            Look closer.

            <br />

            <span>Question what you see.</span>

          </h1>

          <p>

            Analyze images and videos for model-detected signs of AI

            generation or manipulation. Explore face-level predictions

            and sampled video frames.

          </p>

          <div className="hero-note">

            <Info size={16} />

            This is an experimental fake-vs-real estimator, not proof of authenticity.

          </div>

        </section>

        <section className="workspace">

          <div className="panel upload-panel">

            <div className="section-heading">

              <div>

                <p className="eyebrow">START AN ANALYSIS</p>

                <h2>Upload media</h2>

              </div>

              <span className="step-number">01</span>

            </div>

            <div className="mode-tabs">

              <button

                type="button"

                className={mode === "image" ? "mode-tab active" : "mode-tab"}

                onClick={() => changeMode("image")}

              >

                <ImageIcon size={17} />

                Image

              </button>

              <button

                type="button"

                className={mode === "video" ? "mode-tab active" : "mode-tab"}

                onClick={() => changeMode("video")}

              >

                <Film size={17} />

                Video

              </button>

            </div>

            <label

              className={`dropzone ${dragging ? "dragging" : ""} ${

                file ? "has-file" : ""

              }`}

              onDragOver={(event) => {

                event.preventDefault();

                setDragging(true);

              }}

              onDragLeave={() => setDragging(false)}

              onDrop={handleDrop}

            >

              <input

                type="file"

                accept={accepted}

                onChange={(event) => {

                  setSelectedFile(event.target.files?.[0]);

                  event.target.value = "";

                }}

              />

              {preview ? (

                <img

                  className="preview-image"

                  src={preview}

                  alt="Selected image preview"

                />

              ) : (

                <div className="upload-icon">

                  {mode === "image" ? (

                    <UploadCloud size={28} />

                  ) : (

                    <Film size={28} />

                  )}

                </div>

              )}

              <strong>{file ? file.name : "Drop your file here"}</strong>

              <span>

                {file

                  ? `${formatFileSize(file.size)} · Click to replace`

                  : "or click to browse files"}

              </span>

              <small>{fileLabel}</small>

            </label>

            {mode === "video" && (

              <div className="frames-control">

                <label htmlFor="frames">

                  Frames to sample <strong>{maxFrames}</strong>

                </label>

                <input

                  id="frames"

                  type="range"

                  min="4"

                  max="24"

                  step="4"

                  value={maxFrames}

                  onChange={(event) =>

                    setMaxFrames(Number(event.target.value))

                  }

                />

                <small>

                  Frames are sampled across the video. More frames may

                  take longer and use more memory.

                </small>

              </div>

            )}

            {error && (

              <div className="error-message" role="alert">

                <AlertTriangle size={16} />

                <span>{error}</span>

              </div>

            )}

            <button

              type="button"

              className="button primary analyze-button"

              disabled={!file || loading}

              onClick={analyze}

            >

              {loading ? (

                <>

                  <LoaderCircle className="spin" size={18} />

                  Analyzing media…

                </>

              ) : (

                <>

                  <Sparkles size={18} />

                  Analyze {mode === "image" ? "image" : "video"}

                </>

              )}

            </button>

            <p className="privacy-note">

              <ShieldCheck size={15} />

              No account or database. This frontend does not maintain

              analysis history.

            </p>

          </div>

          <div className="panel how-panel">

            <p className="eyebrow">WHAT GETS CHECKED</p>

            <h2>Separate analysis signals</h2>

            <div className="check-item">

              <span className="check-icon">

                <ImageIcon size={18} />

              </span>

              <div>

                <strong>AI-image classification</strong>

                <p>

                  A pretrained model scores the complete image for its

                  learned classes.

                </p>

              </div>

            </div>

            <div className="check-item">

              <span className="check-icon">

                <ScanFace size={18} />

              </span>

              <div>

                <strong>Face-crop fake-vs-real estimate</strong>

                <p>

                  Detected face crops receive a separate AI-generated-vs-real
                  estimate. This is not a validated face-swap detector.

                </p>

              </div>

            </div>

            <div className="check-item">

              <span className="check-icon">

                <Film size={18} />

              </span>

              <div>

                <strong>Video frame sampling</strong>

                <p>

                  Selected frames are analyzed individually. This is

                  not full temporal video deepfake detection.

                </p>

              </div>

            </div>

            <div className="notice-card">

              <AlertTriangle size={19} />

              <p>

                <strong>Know the limits</strong>

                <br />

                Models can disagree. A face-level prediction does not

                prove a face swap, and the system does not currently

                provide a validated manipulation-localization heatmap.

              </p>

            </div>

          </div>

        </section>

        <ResultPanel

          result={result}

          mediaPreview={mode === "image" ? preview : ""}

          videoThumbnail={videoThumbnail}

          onClear={() => {

            setResult(null);

            setError("");

          }}

        />

        <section className="bottom-strip">

          <div>

            <CheckCircle2 size={18} />

            <span>No account required</span>

          </div>

          <div>

            <CheckCircle2 size={18} />

            <span>Open-source stack</span>

          </div>

          <div>

            <CheckCircle2 size={18} />

            <span>JSON + PDF report export</span>

          </div>

        </section>
          </>
        )}
      </main>

      <footer>

        <span>MediaForensicsAI · AI-Powered Media Analysis</span>

        <span>Predictions are estimates, not verification.</span>

      </footer>

    </div>

  );

}
