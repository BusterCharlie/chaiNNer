"use strict";
// Tag Review client. No dependencies, no inline HTML from data: every string
// from the server goes through textContent.

const view = document.getElementById("view");
const toastEl = document.getElementById("toast");
const topbar = document.querySelector(".topbar");

const KNOWN_CATS = new Set([
  "artist", "copyright", "character", "species", "general", "meta", "lore", "rating", "unknown",
]);
const MODE_HINT = {
  verify: "Tap any tag that is wrong",
  random: "Tap any tag that is wrong",
  missing: "Tap tags that ARE in the picture",
};

const state = {
  me: null,
  queued: null, // prefetched next payload
  browse: null, // {filter, tag, items, next, scroll}
  keyHandler: null,
};

// ---------------------------------------------------------------------------
// helpers

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function api(path, { method = "GET", body } = {}) {
  const opts = { method, headers: { "X-DSR": "1" }, credentials: "same-origin" };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch {
    throw new ApiError(0, "Network error, check your connection");
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* empty */
  }
  if (!res.ok) {
    if (res.status === 401 && path !== "/api/login") {
      showLogin("Your session ended. Open your invite link again.");
    }
    throw new ApiError(res.status, (data && data.error) || `Error ${res.status}`);
  }
  return data;
}

let toastTimer = 0;
function toast(msg, { error = false, action = null } = {}) {
  clearTimeout(toastTimer);
  toastEl.replaceChildren(h("span", { text: msg }));
  if (action) {
    toastEl.append(
      h("button", {
        type: "button",
        text: action.label,
        onclick: () => {
          toastEl.hidden = true;
          action.run();
        },
      }),
    );
  }
  toastEl.classList.toggle("error", error);
  toastEl.hidden = false;
  toastTimer = setTimeout(() => (toastEl.hidden = true), action ? 4000 : 2200);
}

function fail(e) {
  if (e instanceof ApiError && e.status === 401) return;
  console.error(e);
  toast(e.message || "Something went wrong", { error: true });
}

function catColor(cat) {
  if (KNOWN_CATS.has(cat)) return `var(--cat-${cat})`;
  let hash = 0;
  for (const ch of cat) hash = (hash * 31 + ch.charCodeAt(0)) | 0;
  return `hsl(${Math.abs(hash) % 360} 65% 60%)`;
}

function dot(cat) {
  const d = h("span", { class: "dot", title: cat });
  d.style.setProperty("--cat", catColor(cat));
  return d;
}

function pct(score) {
  return score === null || score === undefined ? "" : `${Math.round(score * 100)}%`;
}

function tagBudget() {
  const perRow = Math.max(2, Math.floor(innerWidth / 130));
  const rows = Math.max(3, Math.floor((innerHeight * 0.3) / 44));
  return Math.max(6, Math.min(perRow * rows, innerWidth < 600 ? 10 : 20));
}

function setKeys(fn) {
  state.keyHandler = fn;
}
document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (document.querySelector(".sheet-backdrop, .zoom")) return;
  if (state.keyHandler) state.keyHandler(e);
});

function setView(...nodes) {
  setKeys(null);
  view.replaceChildren(...nodes);
}

// ---------------------------------------------------------------------------
// login

function showLogin(message) {
  topbar.hidden = true;
  const input = h("input", {
    class: "field",
    type: "password",
    autocomplete: "off",
    placeholder: "Paste your invite link",
    "aria-label": "Invite link",
  });
  const form = h(
    "form",
    {
      onsubmit: async (e) => {
        e.preventDefault();
        const m = input.value.trim().match(/(?:#key=)?([A-Za-z0-9_-]{20,200})\s*$/);
        if (!m) return toast("That doesn't look like an invite link", { error: true });
        await doLogin(m[1]);
      },
    },
    input,
    h("button", { class: "btn primary", type: "submit", text: "Sign in" }),
  );
  setView(
    h(
      "section",
      { class: "login" },
      h("h1", { text: "Tag Review" }),
      h("p", {
        text: message || "Open the invite link the dataset owner sent you, or paste it here.",
      }),
      form,
    ),
  );
}

async function doLogin(key) {
  try {
    await api("/api/login", { method: "POST", body: { key } });
  } catch (e) {
    showLogin(e.message);
    return;
  }
  await start();
}

// ---------------------------------------------------------------------------
// routing

function route() {
  const hash = location.hash.replace(/^#\/?/, "");
  const [name, arg] = hash.split("/");
  for (const b of document.querySelectorAll("[data-route]")) {
    const active = name === "admin" ? "me" : name || "verify";
    if (b.dataset.route === active) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  }
  switch (name || "verify") {
    case "verify":
    case "missing":
    case "random":
      return showReview(name || "verify").catch(fail);
    case "browse":
      return showBrowse().catch(fail);
    case "image":
      return showDetail(Number(arg)).catch(fail);
    case "admin":
      return showAdmin(arg || "contested").catch(fail);
    case "me":
      return showMe().catch(fail);
    default:
      location.hash = "#/verify";
  }
}

for (const b of document.querySelectorAll("[data-route]")) {
  b.addEventListener("click", () => {
    location.hash = `#/${b.dataset.route}`;
  });
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------------------
// review (verify / missing / random)

async function fetchNext(mode, exclude = []) {
  const q = new URLSearchParams({ mode, n: String(tagBudget()) });
  if (exclude.length) q.set("exclude", exclude.join(","));
  return api(`/api/next?${q}`);
}

function preload(payload) {
  if (payload && payload.image) new Image().src = payload.image.url;
}

async function showReview(mode) {
  setView(h("div", { class: "review" }, h("div", { class: "stage" }, h("div", { class: "loading", text: "Loading…" }))));
  let payload = state.queued && state.queued.mode === mode ? state.queued : null;
  state.queued = null;
  if (!payload) payload = await fetchNext(mode);
  if (currentRoute() !== mode) return;
  renderReview(mode, payload);
  if (!payload.done) {
    fetchNext(mode, [payload.image.id])
      .then((next) => {
        if (!next.done) {
          state.queued = next;
          preload(next);
        }
      })
      .catch(() => {});
  }
}

function currentRoute() {
  return location.hash.replace(/^#\/?/, "").split("/")[0] || "verify";
}

function renderReview(mode, payload) {
  if (payload.done) {
    const others = ["verify", "missing", "random"].filter((m) => m !== mode);
    setView(
      h(
        "section",
        { class: "empty" },
        h("h2", { text: "All caught up here" }),
        h("p", { text: "Nothing left for you in this mode right now. Thanks!" }),
        ...others.map((m) =>
          h("button", { class: "btn", type: "button", text: `Try ${m}`, onclick: () => (location.hash = `#/${m}`) }),
        ),
      ),
    );
    return;
  }

  const img = payload.image;
  const chips = [];
  const chipBox = h("div", { class: "chips", role: "group", "aria-label": "Tags" });

  const cycle = mode === "missing" ? { "-1": 1, 1: 0, 0: -1 } : { 1: -1, "-1": 0, 0: 1 };
  const def = payload.default_vote;

  function makeChip(tag, i) {
    const chip = h(
      "button",
      {
        type: "button",
        class: "chip",
        dataset: { vote: String(def) },
        "aria-pressed": "false",
        title: `${tag.label} (${tag.category})${tag.score !== null ? ` · auto ${pct(tag.score)}` : ""}`,
      },
      dot(tag.category),
      h("span", { class: "label", text: tag.label }),
      tag.status === "contested" ? h("span", { class: "flag", title: "Disputed", text: "⚑" }) : null,
      tag.score !== null ? h("span", { class: "meta", text: pct(tag.score) }) : null,
      i < 10 ? h("span", { class: "key", text: String((i + 1) % 10) }) : null,
    );
    chip.addEventListener("click", () => {
      const v = cycle[chip.dataset.vote];
      chip.dataset.vote = String(v);
      chip.classList.toggle("changed", v !== def);
      chip.setAttribute("aria-pressed", String(v !== def));
    });
    chips.push({ tag, el: chip });
    return chip;
  }

  payload.tags.forEach((t, i) => chipBox.append(makeChip(t, i)));
  if (!payload.tags.length) {
    chipBox.append(h("span", { class: "hint", text: "No tags to check here. Add any that are missing, or skip." }));
  }

  const image = h("img", { src: img.url, alt: "Image under review", decoding: "async" });
  image.addEventListener("click", () => openZoom(img.url));

  let busy = false;
  async function submit(skip) {
    if (busy) return;
    busy = true;
    const votes = skip
      ? []
      : chips
          .map(({ tag, el }) => ({ tag_id: tag.tag_id, vote: Number(el.dataset.vote) }))
          .filter((v) => v.vote !== 0);
    try {
      await api(`/api/image/${img.id}/votes`, { method: "POST", body: { mode, votes } });
    } catch (e) {
      busy = false;
      return fail(e);
    }
    if (!skip) {
      toast("Saved", { action: { label: "Edit", run: () => (location.hash = `#/image/${img.id}`) } });
    }
    showReview(mode).catch(fail);
  }

  function addTag() {
    openAddTag(img.id, (tag) => {
      const chip = h(
        "button",
        { type: "button", class: "chip added", disabled: true, title: "You added this" },
        dot(tag.category),
        h("span", { class: "label", text: tag.label }),
      );
      chipBox.append(chip);
    });
  }

  const submitBtn = h("button", { class: "btn primary", type: "button", text: "Submit", onclick: () => submit(false) });
  setView(
    h(
      "div",
      { class: `review mode-${mode}` },
      h("div", { class: "stage" }, image),
      h("div", { class: "hint" }, h("span", { text: MODE_HINT[mode] })),
      chipBox,
      h(
        "div",
        { class: "actions" },
        h("button", { class: "btn", type: "button", text: "Skip", onclick: () => submit(true) }),
        h("button", { class: "btn", type: "button", text: "+ Tag", onclick: addTag }),
        submitBtn,
      ),
    ),
  );

  setKeys((e) => {
    if (/^[0-9]$/.test(e.key)) {
      const idx = e.key === "0" ? 9 : Number(e.key) - 1;
      if (chips[idx]) chips[idx].el.click();
    } else if (e.key === "Enter") {
      e.preventDefault();
      submit(false);
    } else if (e.key === "s") submit(true);
    else if (e.key === "t" || e.key === "a") {
      e.preventDefault();
      addTag();
    } else if (e.key === "z") openZoom(img.url);
  });
}

// ---------------------------------------------------------------------------
// zoom overlay

function openZoom(url) {
  const img = h("img", { src: url, alt: "" });
  const overlay = h("div", { class: "zoom", role: "dialog", "aria-label": "Zoomed image" }, img);
  const close = () => {
    overlay.remove();
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (e) => {
    if (e.key === "Escape") close();
  };
  let lastTap = 0;
  overlay.addEventListener("click", (e) => {
    const now = Date.now();
    if (e.target === img && now - lastTap > 350) {
      overlay.classList.toggle("full");
    } else if (e.target !== img) {
      close();
    }
    lastTap = now;
  });
  document.addEventListener("keydown", onKey);
  document.body.append(overlay);
}

// ---------------------------------------------------------------------------
// add-tag sheet with autocomplete

function openAddTag(imageId, onAdded) {
  const input = h("input", {
    type: "search",
    placeholder: "Search tags…",
    autocomplete: "off",
    autocapitalize: "off",
    spellcheck: "false",
    enterkeyhint: "done",
    "aria-label": "Tag name",
  });
  const list = h("ul", { class: "results" });
  let results = [];
  let seq = 0;
  let timer = 0;

  const backdrop = h("div", { class: "sheet-backdrop" });
  const close = () => {
    backdrop.remove();
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (e) => {
    if (e.key === "Escape") close();
  };

  async function add(name) {
    try {
      const { tag } = await api(`/api/image/${imageId}/tags`, { method: "POST", body: { name } });
      onAdded(tag);
      toast(`Added ${tag.label}`);
      close();
    } catch (e) {
      fail(e);
    }
  }

  function render() {
    const typed = input.value.trim();
    const exact = results.some((r) => r.label === typed.toLowerCase() || r.name === typed.toLowerCase());
    list.replaceChildren(
      ...results.map((r) =>
        h(
          "li",
          {},
          h(
            "button",
            { type: "button", onclick: () => add(r.name) },
            dot(r.category),
            h("span", { text: r.label }),
            r.alias ? h("span", { class: "alias", text: `← ${r.alias}` }) : null,
            h("span", { class: "count", text: r.count ? r.count.toLocaleString() : r.category }),
          ),
        ),
      ),
      typed && !exact
        ? h(
            "li",
            {},
            h(
              "button",
              { type: "button", onclick: () => add(typed) },
              dot("unknown"),
              h("span", { text: `Add “${typed}”` }),
              h("span", { class: "count", text: "new" }),
            ),
          )
        : null,
    );
  }

  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = input.value.trim();
      const my = ++seq;
      if (!q) {
        results = [];
        return render();
      }
      try {
        const data = await api(`/api/tags/search?q=${encodeURIComponent(q)}`);
        if (my === seq) {
          results = data.results;
          render();
        }
      } catch (e) {
        fail(e);
      }
    }, 120);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const first = results[0];
      const typed = input.value.trim();
      if (first) add(first.name);
      else if (typed) add(typed);
    }
  });

  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) close();
  });
  backdrop.append(
    h("div", { class: "sheet", role: "dialog", "aria-label": "Add a tag" }, input, list),
  );
  document.addEventListener("keydown", onKey);
  document.body.append(backdrop);
  input.focus();
}

// ---------------------------------------------------------------------------
// browse

async function showBrowse() {
  const b = (state.browse = state.browse || { filter: "needs_review", tag: "", items: [], next: 0, scroll: 0 });
  const grid = h("div", { class: "grid" });
  const sentinel = h("div", { class: "sentinel" });
  let loading = false;

  const select = h(
    "select",
    { class: "field", "aria-label": "Filter" },
    ...[
      ["needs_review", "Needs review"],
      ["contested", "Disputed"],
      ["unreviewed", "Nobody voted yet"],
      ["mine", "I voted on"],
      ["all", "All images"],
    ].map(([v, label]) => h("option", { value: v, selected: v === b.filter, text: label })),
  );
  const tagInput = h("input", {
    class: "field",
    type: "search",
    placeholder: "Filter by tag",
    value: b.tag,
    autocapitalize: "off",
    "aria-label": "Filter by tag",
  });

  function cell(item) {
    return h(
      "button",
      { type: "button", onclick: () => {
        b.scroll = scrollY;
        location.hash = `#/image/${item.id}`;
      } },
      h("img", { src: item.thumb, alt: "", loading: "lazy", decoding: "async" }),
      item.contested ? h("span", { class: "badge", text: `⚑ ${item.contested}` }) : null,
    );
  }

  async function more() {
    if (loading || b.next === null) return;
    loading = true;
    try {
      const q = new URLSearchParams({ filter: b.filter, after: String(b.next || 0) });
      if (b.tag) q.set("tag", b.tag);
      const data = await api(`/api/browse?${q}`);
      b.items.push(...data.items);
      b.next = data.next;
      grid.append(...data.items.map(cell));
      if (!b.items.length) grid.append(h("p", { class: "empty", text: "No images match." }));
    } finally {
      loading = false;
    }
  }

  function reset() {
    b.items = [];
    b.next = 0;
    b.scroll = 0;
    grid.replaceChildren();
    more().catch(fail);
  }
  select.addEventListener("change", () => {
    b.filter = select.value;
    reset();
  });
  let t = 0;
  tagInput.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      b.tag = tagInput.value.trim();
      reset();
    }, 400);
  });

  setView(h("section", {}, h("div", { class: "toolbar" }, select, tagInput), grid, sentinel));
  const obs = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) more().catch(fail);
  }, { rootMargin: "400px" });
  obs.observe(sentinel);

  if (b.items.length) {
    grid.append(...b.items.map(cell));
    requestAnimationFrame(() => scrollTo(0, b.scroll));
  } else {
    await more();
  }
}

// ---------------------------------------------------------------------------
// image detail: every tag, vote individually

async function showDetail(id) {
  if (!Number.isInteger(id) || id <= 0) {
    location.hash = "#/browse";
    return;
  }
  const payload = await api(`/api/image/${id}?mode=detail`);
  const img = payload.image;
  const body = h("div", {});

  function chip(tag) {
    const cls = ["chip", `status-${tag.status}`];
    if (tag.my_vote === 1) cls.push("mine-yes");
    if (tag.my_vote === -1) cls.push("mine-no");
    const votes = tag.yes || tag.no ? `✓${fmt(tag.yes)} ✗${fmt(tag.no)}` : pct(tag.score);
    const el = h(
      "button",
      {
        type: "button",
        class: cls.join(" "),
        title: `${tag.label}: ${tag.status.replace("_", " ")}${tag.in_caption ? ", in caption" : ""}`,
      },
      dot(tag.category),
      h("span", { class: "label", text: tag.label }),
      tag.status === "contested" ? h("span", { class: "flag", text: "⚑" }) : null,
      tag.resolution !== null ? h("span", { class: "meta", text: tag.resolution ? "kept" : "removed" }) : null,
      h("span", { class: "meta", text: votes }),
    );
    el.addEventListener("click", async () => {
      const next = tag.my_vote === null ? 1 : tag.my_vote === 1 ? -1 : 0;
      try {
        await api(`/api/image/${id}/votes`, {
          method: "POST",
          body: { mode: "detail", votes: [{ tag_id: tag.tag_id, vote: next }] },
        });
        await refresh();
      } catch (e) {
        fail(e);
      }
    });
    return el;
  }

  async function refresh() {
    const data = await api(`/api/image/${id}?mode=detail`);
    renderTags(data.tags);
  }

  function renderTags(tags) {
    const groups = new Map();
    for (const t of tags) {
      if (!groups.has(t.category)) groups.set(t.category, []);
      groups.get(t.category).push(t);
    }
    // believed-present first within each category
    const present = (t) =>
      t.resolution === 1 || ["confirmed", "leaning_yes", "contested"].includes(t.status) || (t.status === "unreviewed" && t.in_caption);
    body.replaceChildren(
      ...[...groups].flatMap(([cat, list]) => [
        h("h3", {}, dot(cat), cat),
        h("div", { class: "chips" }, ...list.filter(present).map(chip), ...list.filter((t) => !present(t)).map(chip)),
      ]),
    );
  }

  const image = h("img", { src: img.url, alt: "Image" });
  image.addEventListener("click", () => openZoom(img.url));
  setView(
    h(
      "section",
      { class: "detail" },
      h("div", { class: "stage" }, image),
      h(
        "div",
        { class: "bar" },
        h("button", { class: "btn", type: "button", text: "← Back", onclick: () => history.back() }),
        h("button", { class: "btn", type: "button", text: "+ Tag", onclick: () => openAddTag(id, () => refresh().catch(fail)) }),
      ),
      img.path ? h("p", { class: "legend", text: img.path }) : null,
      h("p", {
        class: "legend",
        text: "Tap a tag to vote: yes → no → clear. Your votes are ringed green/red. Struck-through tags were rejected; % is the auto-tagger's confidence.",
      }),
      body,
    ),
  );
  renderTags(payload.tags);
}

function fmt(n) {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

// ---------------------------------------------------------------------------
// admin

async function showAdmin(tab) {
  if (!state.me || state.me.role !== "admin") {
    location.hash = "#/verify";
    return;
  }
  const tabs = h(
    "div",
    { class: "toolbar" },
    h("button", { class: "btn", type: "button", "aria-pressed": String(tab === "contested"), text: "Disputed tags", onclick: () => (location.hash = "#/admin/contested") }),
    h("button", { class: "btn", type: "button", "aria-pressed": String(tab === "categories"), text: "Uncategorised", onclick: () => (location.hash = "#/admin/categories") }),
  );
  if (tab === "categories") return showCategories(tabs);

  const list = h("ul", { class: "list" });
  const moreBtn = h("button", { class: "btn", type: "button", text: "Load more", hidden: true });
  let cursor = [0, 0];

  async function load() {
    const q = new URLSearchParams({ after_image: String(cursor[0]), after_tag: String(cursor[1]) });
    const data = await api(`/api/admin/contested?${q}`);
    for (const item of data.items) list.append(row(item));
    if (!list.children.length) list.append(h("li", { class: "sub", text: "No disputed tags. 🎉" }));
    cursor = data.next || cursor;
    moreBtn.hidden = !data.next;
  }

  function row(item) {
    const li = h(
      "li",
      {},
      h("img", { src: item.thumb, alt: "", loading: "lazy", onclick: () => (location.hash = `#/image/${item.image_id}`) }),
      h(
        "div",
        { class: "grow" },
        h("div", {}, dot(item.category), " ", item.label),
        h("div", {
          class: "sub",
          text: `✓${fmt(item.yes)} ✗${fmt(item.no)} · ${item.in_caption ? "in caption" : "not in caption"}${item.score !== null ? ` · auto ${pct(item.score)}` : ""}`,
        }),
        h("div", { class: "sub", text: item.voters.map((v) => `${v.name} ${v.vote > 0 ? "✓" : "✗"}`).join(", ") }),
      ),
      h("button", { class: "btn good", type: "button", text: "Keep", onclick: () => resolve(1) }),
      h("button", { class: "btn danger", type: "button", text: "Remove", onclick: () => resolve(0) }),
    );
    async function resolve(resolution) {
      try {
        await api("/api/admin/resolve", {
          method: "POST",
          body: { image_id: item.image_id, tag_id: item.tag_id, resolution },
        });
        li.remove();
        toast(resolution ? `Kept ${item.label}` : `Removed ${item.label}`);
      } catch (e) {
        fail(e);
      }
    }
    return li;
  }

  moreBtn.addEventListener("click", () => load().catch(fail));
  setView(h("section", {}, tabs, list, h("div", { class: "toolbar" }, moreBtn)));
  await load();
}

async function showCategories(tabs) {
  const data = await api("/api/admin/uncategorized");
  const cats = state.me.categories.filter((c) => c !== "unknown");
  const list = h("ul", { class: "list" });
  for (const t of data.tags) {
    const sel = h(
      "select",
      { class: "field", "aria-label": `Category for ${t.label}` },
      h("option", { value: "", text: "Choose…" }),
      ...cats.map((c) => h("option", { value: c, text: c })),
      h("option", { value: "__new", text: "New category…" }),
    );
    const li = h(
      "li",
      {},
      h("div", { class: "grow" }, h("div", { text: t.label }), h("div", { class: "sub", text: `${t.uses} images` })),
      sel,
    );
    sel.addEventListener("change", async () => {
      let cat = sel.value;
      if (cat === "__new") cat = (prompt("New category name") || "").trim();
      if (!cat) return;
      try {
        await api("/api/admin/category", { method: "POST", body: { tag: t.name, category: cat } });
        if (!state.me.categories.includes(cat)) state.me.categories.push(cat);
        li.remove();
        toast(`${t.label} → ${cat}`);
      } catch (e) {
        fail(e);
      }
    });
    list.append(li);
  }
  if (!data.tags.length) list.append(h("li", { class: "sub", text: "Every tag in use has a category." }));
  setView(h("section", {}, tabs, list));
}

// ---------------------------------------------------------------------------
// me / stats

async function showMe() {
  const s = await api("/api/stats");
  const done = s.images ? 1 - s.images_needing_review / s.images : 0;
  const bar = h("div", {});
  bar.style.width = `${Math.round(done * 100)}%`;
  const statuses = Object.entries(s.tags_by_status).sort((a, b) => b[1] - a[1]);
  setView(
    h(
      "section",
      {},
      h("h2", { class: "section-title", text: `Hi ${state.me.name}` }),
      state.me.role === "admin"
        ? h(
            "div",
            { class: "toolbar" },
            h("button", { class: "btn", type: "button", text: "⚑ Disputed tags", onclick: () => (location.hash = "#/admin/contested") }),
            h("button", { class: "btn", type: "button", text: "Uncategorised tags", onclick: () => (location.hash = "#/admin/categories") }),
          )
        : null,
      h(
        "div",
        { class: "stat-row" },
        h("div", { class: "stat" }, h("b", { text: s.my_votes.toLocaleString() }), h("span", { text: "your votes" })),
        h("div", { class: "stat" }, h("b", { text: s.images_needing_review.toLocaleString() }), h("span", { text: `of ${s.images.toLocaleString()} images still need review` })),
      ),
      h("div", { class: "progress", title: `${Math.round(done * 100)}% reviewed` }, bar),
      h("h2", { class: "section-title", text: "Tag status" }),
      h(
        "ul",
        { class: "list" },
        ...statuses.map(([k, v]) => h("li", {}, h("span", { class: "grow", text: k.replace("_", " ") }), h("b", { text: v.toLocaleString() }))),
      ),
      h("h2", { class: "section-title", text: "Leaderboard" }),
      h(
        "ul",
        { class: "list" },
        ...s.leaderboard.map((r) => h("li", {}, h("span", { class: "grow", text: r.name }), h("b", { text: r.votes.toLocaleString() }))),
      ),
      h(
        "div",
        { class: "toolbar" },
        h("button", {
          class: "btn danger",
          type: "button",
          text: "Sign out on this device",
          onclick: async () => {
            await api("/api/logout", { method: "POST", body: {} }).catch(() => {});
            showLogin("Signed out.");
          },
        }),
      ),
    ),
  );
}

// ---------------------------------------------------------------------------
// boot

async function start() {
  try {
    state.me = await api("/api/me");
  } catch (e) {
    if (e.status === 401) return showLogin();
    return fail(e);
  }
  document.title = state.me.title;
  topbar.hidden = false;
  document.querySelector(".me").textContent = state.me.name.slice(0, 1).toUpperCase();
  if (!location.hash || location.hash.startsWith("#key=")) {
    history.replaceState(null, "", `${location.pathname}#/verify`);
  }
  route();
}

(async function boot() {
  const m = location.hash.match(/^#key=([A-Za-z0-9_-]{20,200})$/);
  if (m) {
    // Remove the key from the address bar and history before anything else.
    history.replaceState(null, "", location.pathname);
    await doLogin(m[1]);
    return;
  }
  await start();
})();
