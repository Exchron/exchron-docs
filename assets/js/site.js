(() => {
  const root = document.documentElement;
  const themeToggle = document.querySelector("[data-theme-toggle]");
  const menuToggle = document.querySelector(".navbar__toggle");
  const sidebar = document.querySelector(".docSidebarContainer_YfHR");
  const backToTop = document.querySelector(".theme-back-to-top-button");

  function readThemePreference() {
    try {
      return localStorage.getItem("exchron-theme");
    } catch (error) {
      if (error instanceof DOMException && error.name === "SecurityError") {
        return null;
      }
      throw error;
    }
  }

  function writeThemePreference(theme) {
    try {
      localStorage.setItem("exchron-theme", theme);
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "SecurityError")) {
        throw error;
      }
    }
  }

  function applyTheme(theme, persist = false) {
    root.dataset.theme = theme;
    if (themeToggle) {
      const nextTheme = theme === "dark" ? "light" : "dark";
      themeToggle.setAttribute("aria-label", `Switch to ${nextTheme} mode`);
      themeToggle.title = `Switch to ${nextTheme} mode`;
    }
    if (persist) {
      writeThemePreference(theme);
    }
  }

  const savedTheme = readThemePreference();
  applyTheme(savedTheme === "dark" ? "dark" : "light");

  themeToggle?.addEventListener("click", () => {
    applyTheme(root.dataset.theme === "dark" ? "light" : "dark", true);
  });

  document.querySelectorAll(".tocCollapsibleButton_TO0P").forEach((button) => {
    const container = button.closest(".tocCollapsible_ETCw");
    const contents = container?.querySelector(".tocCollapsibleContent_vkbj");
    if (!container || !contents) {
      return;
    }

    button.setAttribute("aria-expanded", "false");
    contents.hidden = true;
    button.addEventListener("click", () => {
      const expanded = button.getAttribute("aria-expanded") !== "true";
      button.setAttribute("aria-expanded", String(expanded));
      button.classList.toggle("tocCollapsibleButtonExpanded_MG3E", expanded);
      container.classList.toggle("tocCollapsibleExpanded_sAul", expanded);
      contents.hidden = !expanded;
    });
  });

  document.querySelectorAll('[role="tablist"]').forEach((tabList, tabListIndex) => {
    const tabs = Array.from(tabList.querySelectorAll('[role="tab"]'));
    const panels = Array.from(
      tabList.parentElement.querySelectorAll('[role="tabpanel"]'),
    );
    if (tabs.length === 0 || tabs.length !== panels.length) {
      return;
    }

    function selectTab(index, moveFocus = false) {
      tabs.forEach((tab, tabIndex) => {
        const selected = tabIndex === index;
        tab.setAttribute("aria-selected", String(selected));
        tab.setAttribute("tabindex", selected ? "0" : "-1");
        tab.classList.toggle("tabs__item--active", selected);
        panels[tabIndex].hidden = !selected;
      });
      if (moveFocus) {
        tabs[index].focus();
      }
    }

    const initialIndex = Math.max(
      0,
      tabs.findIndex((tab) => tab.getAttribute("aria-selected") === "true"),
    );
    panels.forEach((panel, index) => {
      panel.id = `plain-tabpanel-${tabListIndex}-${index}`;
      panel.tabIndex = 0;
      tabs[index].setAttribute("aria-controls", panel.id);
      tabs[index].addEventListener("click", () => selectTab(index));
      tabs[index].addEventListener("keydown", (event) => {
        let nextIndex = index;
        if (event.key === "ArrowRight") {
          nextIndex = (index + 1) % tabs.length;
        } else if (event.key === "ArrowLeft") {
          nextIndex = (index - 1 + tabs.length) % tabs.length;
        } else if (event.key === "Home") {
          nextIndex = 0;
        } else if (event.key === "End") {
          nextIndex = tabs.length - 1;
        } else {
          return;
        }
        event.preventDefault();
        selectTab(nextIndex, true);
      });
    });
    selectTab(initialIndex);
  });

  if (backToTop) {
    const updateBackToTop = () => {
      backToTop.classList.toggle("backToTopButtonShow_xfvO", window.scrollY > 300);
    };

    window.addEventListener("scroll", updateBackToTop, { passive: true });
    backToTop.addEventListener("click", () => {
      window.scrollTo({
        top: 0,
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
      });
    });
    updateBackToTop();
  }

  if (!menuToggle || !sidebar) {
    return;
  }

  const backdrop = document.createElement("button");
  backdrop.className = "plain-menu-backdrop";
  backdrop.type = "button";
  backdrop.setAttribute("aria-label", "Close navigation");
  document.body.append(backdrop);

  function setMenuOpen(open) {
    document.body.classList.toggle("plain-menu-open", open);
    menuToggle.setAttribute("aria-expanded", String(open));
  }

  menuToggle.addEventListener("click", () => {
    setMenuOpen(menuToggle.getAttribute("aria-expanded") !== "true");
  });

  backdrop.addEventListener("click", () => setMenuOpen(false));

  sidebar.addEventListener("click", (event) => {
    const link = event.target.closest('a[role="button"]');
    if (link) {
      const item = link.closest(".menu__list-item");
      const children = item?.querySelector(":scope > ul");
      if (children) {
        event.preventDefault();
        const expanded = link.getAttribute("aria-expanded") !== "true";
        link.setAttribute("aria-expanded", String(expanded));
        item.classList.toggle("menu__list-item--collapsed", !expanded);
        return;
      }
    }

    if (event.target.closest("a")) {
      setMenuOpen(false);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("plain-menu-open")) {
      setMenuOpen(false);
      menuToggle.focus();
    }
  });
})();
