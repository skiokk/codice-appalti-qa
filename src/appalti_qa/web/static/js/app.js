// Progressive enhancements: the pages work without this script.
(() => {
  "use strict";

  const ANNOUNCE_DELAY_MS = 800;
  let announceTimer;

  // Character counter for textareas that declare data-contatore="<counter element id>".
  // The visible count updates at once; screen readers hear the remainder after a pause.
  const showCount = (field) => {
    const counter = document.getElementById(field.dataset.contatore);
    if (counter) {
      counter.textContent = `Hai scritto ${field.value.length} caratteri su ${field.maxLength}.`;
    }
  };

  const announceRemaining = (field) => {
    const announcer = document.getElementById(`${field.dataset.contatore}-annuncio`);
    if (!announcer) return;
    const remaining = field.maxLength - field.value.length;
    clearTimeout(announceTimer);
    announceTimer = setTimeout(() => {
      announcer.textContent =
        remaining === 1 ? "Resta 1 carattere." : `Restano ${remaining} caratteri.`;
    }, ANNOUNCE_DELAY_MS);
  };

  const refreshCounters = () =>
    document.querySelectorAll("textarea[data-contatore]").forEach(showCount);

  document.addEventListener("input", (event) => {
    const field = event.target;
    if (field instanceof HTMLTextAreaElement && field.dataset.contatore) {
      showCount(field);
      announceRemaining(field);
    }
  });
  document.addEventListener("DOMContentLoaded", refreshCounters);
  // The form fields are re-rendered by the server after each htmx request.
  document.addEventListener("htmx:afterSettle", refreshCounters);

  // Tell screen reader users that the question was sent while the answer is prepared.
  const setStatus = (text) => {
    const status = document.getElementById("stato-richiesta");
    if (status) status.textContent = text;
  };
  document.addEventListener("htmx:beforeRequest", () =>
    setStatus("Domanda inviata: ricerca delle fonti e preparazione della risposta in corso."),
  );
  document.addEventListener("htmx:afterRequest", () => setStatus(""));
})();
