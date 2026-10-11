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
  // The region being updated is marked busy while the request runs.
  const targetOf = (event) => event.detail.target || document.getElementById("esito");

  document.addEventListener("htmx:beforeRequest", (event) => {
    targetOf(event)?.setAttribute("aria-busy", "true");
    setStatus("Domanda inviata: ricerca delle fonti e preparazione della risposta in corso.");
  });
  document.addEventListener("htmx:afterRequest", (event) => {
    targetOf(event)?.removeAttribute("aria-busy");
    setStatus("");
  });

  // Failures that htmx does not swap (a status the server did not render, a network error, a
  // timeout) would otherwise leave the page unchanged: show and announce an error instead.
  const ERROR_MESSAGES = {
    response: (status) =>
      `Non è stato possibile ottenere la risposta (errore ${status}). Riprova tra qualche ` +
      "minuto; se il problema persiste, consulta il testo ufficiale su Normattiva.",
    send: () =>
      "Non è stato possibile contattare il servizio. Controlla la connessione a Internet e riprova.",
    timeout: () =>
      "Il servizio non ha risposto in tempo. Riprova tra qualche minuto; se il problema " +
      "persiste, consulta il testo ufficiale su Normattiva.",
  };

  // Undo what the request turned on, in case htmx stopped before doing it itself.
  const resetRequestState = (event) => {
    const source = event.detail.elt;
    targetOf(event)?.removeAttribute("aria-busy");
    setStatus("");
    if (!(source instanceof Element)) return;
    source.classList.remove("htmx-request");
    const indicator = source.getAttribute("hx-indicator");
    if (indicator) document.querySelector(indicator)?.classList.remove("htmx-request");
    source.querySelectorAll("button[type='submit'][disabled]").forEach((button) => {
      button.disabled = false;
    });
  };

  const showError = (event, message) => {
    resetRequestState(event);
    const region = targetOf(event);
    if (!region) return;
    const alert = document.createElement("div");
    alert.className = "alert alert-danger";
    alert.setAttribute("role", "alert");
    const title = document.createElement("h2");
    title.className = "h5";
    title.textContent = "Servizio non disponibile";
    const text = document.createElement("p");
    text.className = "mb-0";
    text.textContent = message;
    alert.append(title, text);
    region.replaceChildren(alert);
  };

  document.addEventListener("htmx:responseError", (event) =>
    showError(event, ERROR_MESSAGES.response(event.detail.xhr?.status ?? "sconosciuto")),
  );
  document.addEventListener("htmx:sendError", (event) =>
    showError(event, ERROR_MESSAGES.send()),
  );
  document.addEventListener("htmx:timeout", (event) =>
    showError(event, ERROR_MESSAGES.timeout()),
  );
})();
