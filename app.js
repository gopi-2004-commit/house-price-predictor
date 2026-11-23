const $ = id => document.getElementById(id);
const FIELDS = ["locality", "area_sqft", "bedrooms", "bathrooms", "age_years", "furnishing", "parking"];
const DEFAULTS = { area_sqft: 1100, bedrooms: 2, bathrooms: 2, age_years: 8, furnishing: 1, parking: 1 };
let meta;

const inr = n => n >= 1e7 ? "₹" + (n / 1e7).toFixed(2) + " Cr" : "₹" + (n / 1e5).toFixed(n >= 1e6 ? 1 : 2) + " L";
const signed = n => (n < 0 ? "−" : "+") + inr(Math.abs(n));
const showError = m => { $("error").textContent = m; $("error").hidden = !m; };

async function api(path, body) {
  const res = await fetch("/api" + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Request failed (" + res.status + ")");
  return data;
}

function fillDefaults() {
  FIELDS.slice(1).forEach(k => { $(k).value = DEFAULTS[k]; });
  $("locality").value = meta.localities.includes("Salt Lake") ? "Salt Lake" : meta.localities[0];
  showError("");
}

function collect() {
  const body = { locality: $("locality").value };
  for (const k of FIELDS.slice(1)) {
    const raw = $(k).value.trim();
    if (raw === "" || isNaN(+raw)) throw new Error("Enter a number for " + $("label_" + k).textContent + ".");
    body[k] = +raw;
    const lim = meta.limits[k];
    if (lim && (body[k] < lim[0] || body[k] > lim[1]))
      throw new Error($("label_" + k).textContent + " must be between " + lim[0] + " and " + lim[1] + ".");
  }
  return body;
}

function bars(items, vKey, nKey, fmt, isSigned, mark) {
  const max = Math.max(...items.map(i => Math.abs(i[vKey]))) || 1;
  return items.map(i => {
    const v = i[vKey], w = Math.abs(v) / max * (isSigned ? 50 : 100);
    const style = isSigned
      ? `${v >= 0 ? "left:50%" : "right:50%"};width:${w}%;background:${v >= 0 ? "var(--accent)" : "var(--neg)"}`
      : `left:0;width:${w}%;background:${mark(i) ? "var(--warm)" : "var(--accent)"}`;
    return `<div class="row${!isSigned && mark(i) ? " me" : ""}"><span class="n">${i[nKey]}</span><div class="track"><span style="${style}"></span></div><span class="v">${fmt(v)}</span></div>`;
  }).join("");
}

function render(r, body) {
  const d = r.vs_average_pct;
  $("out").innerHTML = `
    <section class="result">
      <div>Estimated price</div>
      <div class="price">${inr(r.price)}</div>
      <div class="range">Likely range (80%): ${inr(r.low)} to ${inr(r.high)}</div>
      <div class="meta">
        <div><b>₹${Math.round(r.per_sqft).toLocaleString("en-IN")}</b>per sq ft</div>
        <div><b>${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(0)}%</b>vs. average home</div>
      </div>
    </section>
    <div class="two">
      <section class="panel"><h2>What moves the price</h2>${bars(r.factors, "effect", "name", signed, true)}
        <p class="note">Each detail compared with an average home.</p></section>
      <section class="panel"><h2>Same home elsewhere</h2>${bars(r.comparison, "price", "locality", inr, false, i => i.locality === body.locality)}</section>
    </div>`;
}

$("form").addEventListener("submit", async e => {
  e.preventDefault();
  showError("");
  let body;
  try { body = collect(); } catch (err) { showError(err.message); return; }
  const btn = $("go"); btn.disabled = true; btn.textContent = "Predicting…";
  try { render(await api("/predict", body), body); }
  catch (err) { showError("Could not get a prediction. " + err.message); }
  finally { btn.disabled = false; btn.textContent = "Predict price"; }
});

$("reset").addEventListener("click", () => {
  fillDefaults();
  $("out").innerHTML = '<div class="panel empty">Your estimate will appear here.</div>';
});

(async function init() {
  // give each field a readable name for error messages
  const names = { area_sqft: "Area", bedrooms: "Bedrooms", bathrooms: "Bathrooms", age_years: "Age" };
  Object.entries(names).forEach(([k, n]) => { const s = document.createElement("span"); s.id = "label_" + k; s.hidden = true; s.textContent = n; document.body.appendChild(s); });
  try { meta = await api("/meta"); }
  catch (err) { showError("Cannot reach the API. Start the backend with: python app.py"); return; }
  $("locality").innerHTML = meta.localities.map(l => `<option>${l}</option>`).join("");
  const L = meta.limits, set = (k, h) => { const [a, b] = L[k]; $(k).min = a; $(k).max = b; $(h).textContent = `(${a}–${b})`; };
  set("area_sqft", "h_area"); set("bedrooms", "h_bed"); set("bathrooms", "h_bath"); set("age_years", "h_age");
  fillDefaults();
})();
