import { useState, type KeyboardEvent, type SubmitEvent } from "react";

interface ComposerProps {
  isStreaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

/** Message input. Enter sends, Shift+Enter inserts a newline. */
export function Composer({ isStreaming, onSend, onStop }: ComposerProps) {
  const [text, setText] = useState("");

  const submit = () => {
    if (!text.trim() || isStreaming) return;
    onSend(text);
    setText("");
  };

  const onSubmit = (e: SubmitEvent<HTMLFormElement>) => {
    e.preventDefault();
    submit();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
        }}
        onKeyDown={onKeyDown}
        placeholder="Message Recall…"
        rows={1}
        maxLength={8000}
        aria-label="Message"
      />
      {isStreaming ? (
        <button type="button" className="btn" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="submit" className="btn btn-primary" disabled={!text.trim()}>
          Send
        </button>
      )}
    </form>
  );
}
