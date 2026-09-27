/*
  SRP Theme Pack Helper
  Debug logs included intentionally to avoid silent failures.
*/
(function () {
  const STORAGE_KEY = "srp-theme-mode";

  function log(...args) {
    console.debug("[SRP_THEME]", ...args);
  }

  function applyTheme(mode) {
    const root = document.documentElement;
    const safeMode = mode === "srp-light" ? "srp-light" : "srp-dark";
    root.setAttribute("data-theme", safeMode);
    try {
      localStorage.setItem(STORAGE_KEY, safeMode);
      log("Theme applied:", safeMode);
    } catch (err) {
      console.warn("[SRP_THEME] Failed to persist theme mode", err);
    }
    return safeMode;
  }

  function getSavedTheme() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      console.warn("[SRP_THEME] Failed to read saved theme", err);
      return null;
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") || "srp-dark";
    const next = current === "srp-dark" ? "srp-light" : "srp-dark";
    return applyTheme(next);
  }

  function initTheme(defaultMode) {
    const saved = getSavedTheme();
    const initial = saved || defaultMode || "srp-dark";
    applyTheme(initial);
    log("Theme initialized:", initial);
  }

  window.SRPTheme = {
    initTheme,
    applyTheme,
    toggleTheme,
  };
})();
