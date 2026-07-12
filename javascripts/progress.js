(() => {
  "use strict";

  const STORAGE_KEY = "inside-the-model-progress-v1";
  const lessons = [
    "modules/00-orientation/",
    "modules/01-transformer-anatomy/",
    "modules/02-circuit-algebra/",
    "modules/03-causal-experiments/",
    "modules/04-canonical-circuits/",
    "modules/05-automated-circuit-discovery/",
    "modules/06-superposition/",
    "modules/07-sparse-autoencoders/",
    "modules/08-transcoders-crosscoders/",
    "modules/09-attribution-graphs/",
    "modules/10-explanations-workspace/",
    "modules/11-steering/",
    "modules/12-safety-mechanisms/",
    "modules/13-auditing/",
    "modules/14-research-rigor/",
    "modules/15-project-design/",
    "labs/00-environment/",
    "labs/01-activation-cache/",
    "labs/02-activation-patching/",
    "labs/03-sae-features/",
    "labs/04-circuit-tracer/",
    "labs/05-jacobian-lens/",
    "labs/06-safety-audit/",
    "labs/07-explanation-showdown/",
    "labs/08-capstone/",
  ];

  function readProgress() {
    try {
      const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return new Set(Array.isArray(value) ? value.filter((item) => lessons.includes(item)) : []);
    } catch (_) {
      return new Set();
    }
  }

  function writeProgress(progress) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...progress]));
  }

  function currentLesson() {
    const path = window.location.pathname.replace(/^.*ai-safety-interpretability-circuits\//, "");
    return lessons.find((lesson) => path.endsWith(lesson)) || null;
  }

  function updateDashboard(progress) {
    const count = progress.size;
    const percent = Math.round((count / lessons.length) * 100);
    document.querySelectorAll("[data-course-count]").forEach((node) => {
      node.textContent = `${count} / ${lessons.length} lessons complete`;
    });
    document.querySelectorAll("[data-course-progress-bar]").forEach((node) => {
      node.style.width = `${percent}%`;
    });
    document.querySelectorAll(".course-progress-track").forEach((node) => {
      node.setAttribute("aria-valuenow", String(count));
      node.setAttribute("aria-valuemax", String(lessons.length));
    });
  }

  function updateNavigation(progress) {
    document.querySelectorAll(".md-nav__link[href]").forEach((link) => {
      const href = new URL(link.href, window.location.href).pathname;
      const lesson = lessons.find((candidate) => href.endsWith(candidate));
      link.classList.toggle("course-done", Boolean(lesson && progress.has(lesson)));
    });
  }

  function installCompletionButton(progress) {
    const lesson = currentLesson();
    const article = document.querySelector("article.md-content__inner");
    const heading = article?.querySelector("h1");
    if (!lesson || !heading || article.querySelector(".course-complete")) return;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "course-complete";

    const render = () => {
      const complete = progress.has(lesson);
      button.setAttribute("aria-pressed", String(complete));
      button.textContent = complete ? "✓ Completed" : "Mark complete";
    };

    button.addEventListener("click", () => {
      if (progress.has(lesson)) progress.delete(lesson);
      else progress.add(lesson);
      writeProgress(progress);
      render();
      updateDashboard(progress);
      updateNavigation(progress);
    });
    render();
    heading.insertAdjacentElement("afterend", button);
  }

  function installReset(progress) {
    document.querySelectorAll("[data-course-reset]").forEach((button) => {
      button.addEventListener("click", () => {
        if (!window.confirm("Reset course progress stored in this browser?")) return;
        progress.clear();
        writeProgress(progress);
        updateDashboard(progress);
        updateNavigation(progress);
      });
    });
  }

  function initialize() {
    const progress = readProgress();
    updateDashboard(progress);
    updateNavigation(progress);
    installCompletionButton(progress);
    installReset(progress);
  }

  if (typeof document$ !== "undefined") document$.subscribe(initialize);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();
})();

