/**
 * XLA Face Privacy System — Shared helpers
 * Included in every page via <script src="/ui/assets/js/xla.js">
 */

const BASE = 'http://localhost:8000';

// ─── Session / Auth ──────────────────────────────────────────────────────────
function getToken()  { return sessionStorage.getItem('xla_token'); }
function getRole()   { return sessionStorage.getItem('xla_role'); }
function setSession(token, role, expires) {
  sessionStorage.setItem('xla_token', token || '');
  sessionStorage.setItem('xla_role', role);
  if (expires) sessionStorage.setItem('xla_expires', expires);
}
function clearSession() {
  ['xla_token', 'xla_role', 'xla_expires'].forEach(k => sessionStorage.removeItem(k));
}
function guardAdmin() {
  if (getRole() !== 'admin' || !getToken()) { window.location.href = '/'; return false; }
  return true;
}
function guardUser() {
  if (!getRole()) { window.location.href = '/'; return false; }
  return true;
}
async function doLogout() {
  const token = getToken();
  if (token) { try { await apiPost('/admin/logout', { token }); } catch (_) {} }
  clearSession();
  window.location.href = '/';
}

function switchToUser() {
  sessionStorage.setItem('xla_role', 'user');
  window.location.href = '/ui/user/live-portal.html';
}

function switchToAdmin() {
  if (getToken()) {
    sessionStorage.setItem('xla_role', 'admin');
    window.location.href = '/ui/admin/dashboard.html';
  } else {
    window.location.href = '/';
  }
}

// ─── API client ───────────────────────────────────────────────────────────────
async function apiFetch(method, path, body, isForm = false) {
  const opts = { method, headers: isForm ? {} : { 'Content-Type': 'application/json' } };
  if (body != null) opts.body = isForm ? body : JSON.stringify(body);
  const res = await fetch(BASE + path, opts);
  if (!res.ok) {
    let msg;
    try { msg = (await res.json()).detail; } catch (_) { msg = res.statusText; }
    throw new Error(msg || `HTTP ${res.status}`);
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : res;
}
const apiGet    = (path)       => apiFetch('GET',    path);
const apiPost   = (path, body) => apiFetch('POST',   path, body);
const apiPatch  = (path, body) => apiFetch('PATCH',  path, body);
const apiDelete = (path, body) => apiFetch('DELETE', path, body);
const apiForm   = (path, fd)   => apiFetch('POST',   path, fd, true);

// ─── UI Helpers ───────────────────────────────────────────────────────────────
function toast(msg, type = 'ok') {
  const colors = {
    ok:   'bg-green-400 text-black',
    err:  'bg-red-500 text-white',
    info: 'bg-white text-black',
  };
  const el = document.createElement('div');
  el.className = `fixed bottom-6 right-6 z-[9999] px-6 py-4 font-black text-sm
                  uppercase border-4 border-black shadow-[6px_6px_0px_0px_#000]
                  ${colors[type] || colors.info}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

function openModal(id)  { document.getElementById(id)?.classList.remove('hidden'); }
function closeModal(id) { document.getElementById(id)?.classList.add('hidden'); }

function setBtnLoading(btn, on, customText) {
  if (!btn) return;
  if (on) { 
    btn._orig = btn.innerHTML; 
    btn.innerHTML = customText || 'LOADING...'; 
    btn.disabled = true; 
  } else { 
    btn.innerHTML = btn._orig || 'OK'; 
    btn.disabled = false; 
  }
}

// ─── Time helpers ─────────────────────────────────────────────────────────────
function fmtTime(iso) {
  try { return new Date(iso).toLocaleTimeString('vi-VN', { hour12: false }); }
  catch (_) { return iso; }
}
function logLine(msg, color = 'text-primary') {
  const wrap = document.getElementById('log-body');
  if (!wrap) return;
  const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
  const p = document.createElement('p');
  p.className = `flex gap-4 ${color}`;
  p.innerHTML = `<span class="opacity-40 text-[10px] min-w-[58px]">${now}</span><span class="text-white">></span> ${msg}`;
  wrap.prepend(p);
  // Keep at most 50 log lines
  while (wrap.children.length > 50) wrap.removeChild(wrap.lastChild);
}
