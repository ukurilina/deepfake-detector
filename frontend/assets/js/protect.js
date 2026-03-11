import { setStatus } from "./ui.js";

document.addEventListener("DOMContentLoaded", () => {
  const disabledButton = document.getElementById("protectButton");
  if (disabledButton) {
    disabledButton.addEventListener("click", (event) => {
      event.preventDefault();
      setStatus("Эта функция пока недоступна.", "error");
    });
  }
});
