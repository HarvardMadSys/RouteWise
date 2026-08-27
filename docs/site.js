const header = document.querySelector("[data-header]");
const menuButton = document.querySelector("[data-menu-button]");
const menu = document.querySelector("[data-menu]");
const toast = document.querySelector("[data-toast]");

function updateHeader() {
  header?.classList.toggle("scrolled", window.scrollY > 12);
}

function closeMenu() {
  if (!menuButton || !menu) return;
  menuButton.setAttribute("aria-expanded", "false");
  menu.classList.remove("open");
  document.body.classList.remove("menu-open");
}

menuButton?.addEventListener("click", () => {
  if (!menu) return;
  const isOpen = menuButton.getAttribute("aria-expanded") === "true";
  menuButton.setAttribute("aria-expanded", String(!isOpen));
  menu.classList.toggle("open", !isOpen);
  document.body.classList.toggle("menu-open", !isOpen);
});

menu?.querySelectorAll("a").forEach((link) => link.addEventListener("click", closeMenu));

window.addEventListener("scroll", updateHeader, { passive: true });
window.addEventListener("resize", () => {
  if (window.innerWidth > 820) closeMenu();
});
updateHeader();

let toastTimer;
function showToast(message = "Copied to clipboard") {
  if (!toast) return;
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.add("visible");
  toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 1800);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast();
  } catch {
    showToast("Copy unavailable");
  }
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", () => copyText(button.dataset.copy));
});

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (target) copyText(target.textContent.trim());
  });
});

document.querySelectorAll("[data-year]").forEach((element) => {
  element.textContent = new Date().getFullYear();
});
