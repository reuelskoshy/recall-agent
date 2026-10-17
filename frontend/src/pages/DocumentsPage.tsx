import { useRef, useState, type DragEvent } from "react";

import { useDocuments } from "../hooks/useDocuments";
import type { DocumentInfo } from "../types/api";

const ACCEPT = ".pdf,.txt,.md";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${String(bytes)} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function StatusBadge({ doc }: { doc: DocumentInfo }) {
  if (doc.status === "processing") {
    return (
      <span className="badge">
        <span className="spinner" /> Processing
      </span>
    );
  }
  if (doc.status === "failed") return <span className="badge badge-bad">Failed</span>;
  return <span className="badge badge-ok">Ready</span>;
}

/** Upload documents for RAG and watch them get chunked and embedded. */
export function DocumentsPage() {
  const { documents, uploads, error, upload, dismissUpload, remove } = useDocuments();
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    void upload(Array.from(e.dataTransfer.files));
  };

  return (
    <section className="page">
      <h2>Documents</h2>
      <p className="page-intro">
        Upload PDF, TXT, or Markdown files. Each is split into chunks, embedded locally, and
        stored in pgvector. The agent searches them with <code>search_documents</code> and
        cites file and page.
      </p>

      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => {
          setDragging(false);
        }}
        onDrop={onDrop}
      >
        <strong>Drop files here</strong> or click to choose
        <span className="dropzone-hint">PDF, TXT, MD · up to 20 MB</span>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          multiple
          hidden
          onChange={(e) => {
            void upload(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
      </div>

      {uploads.map((u) => (
        <div key={u.name} className={`upload-row ${u.error ? "upload-error" : ""}`}>
          {u.error ? (
            <>
              ⚠ {u.name}: {u.error}
              <button
                type="button"
                className="link-button"
                onClick={() => {
                  dismissUpload(u.name);
                }}
              >
                Dismiss
              </button>
            </>
          ) : (
            <>
              <span className="spinner" /> Uploading {u.name}…
            </>
          )}
        </div>
      ))}

      {error && <p className="message-error">Could not load documents: {error}</p>}

      {documents.length === 0 && !error ? (
        <p className="page-empty">No documents yet.</p>
      ) : (
        <ul className="doc-list">
          {documents.map((doc) => (
            <li key={doc.id} className="doc-row">
              <div className="doc-main">
                <span className="doc-name" title={doc.filename}>
                  {doc.filename}
                </span>
                <span className="doc-meta">
                  {formatSize(doc.size_bytes)}
                  {doc.page_count !== null && ` · ${String(doc.page_count)} pages`}
                  {doc.status === "ready" && ` · ${String(doc.chunk_count)} chunks`}
                </span>
                {doc.error && <span className="doc-error">{doc.error}</span>}
              </div>
              <StatusBadge doc={doc} />
              <button
                type="button"
                className="sidebar-delete doc-delete"
                aria-label={`Delete ${doc.filename}`}
                onClick={() => {
                  if (window.confirm(`Delete "${doc.filename}" and its chunks?`)) {
                    void remove(doc.id);
                  }
                }}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
