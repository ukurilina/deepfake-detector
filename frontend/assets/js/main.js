document.addEventListener("DOMContentLoaded", () => {
  const currentYear = document.getElementById("currentYear");
  if (currentYear) {
    currentYear.textContent = String(new Date().getFullYear());
  }
});

