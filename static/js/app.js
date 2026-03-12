const ta = document.getElementById("text");
const counter = document.getElementById("counter");

function updateCounter(){
  if(!ta || !counter) return;
  counter.textContent = (ta.value || "").length.toString();
}
if(ta){
  ta.addEventListener("input", updateCounter);
  window.addEventListener("load", updateCounter);
}

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("analyzeForm");
  const overlay = document.getElementById("loadingOverlay");
  const predictBtn = document.getElementById("predictBtn");

  if (!form || !overlay) return;

  form.addEventListener("submit", function () {
    overlay.style.display = "flex";
    form.classList.add("form--loading");

    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.textContent = "Идёт анализ...";
    }
  });

  window.addEventListener("pageshow", function () {
    overlay.style.display = "none";
    form.classList.remove("form--loading");

    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.textContent = "Проанализировать текст";
    }
  });
});
