"""Minimal in-browser Admin UI for API key management.

This router serves a single HTML page at ``/admin/ui``. The page prompts for
an admin token (not stored on the server) and uses it in Authorization headers
to call existing admin APIs:

- POST /admin/api-keys
- GET /admin/api-keys
- GET /admin/api-keys/{user_id}
- PATCH /admin/api-keys/{user_id}
- DELETE /admin/api-keys/{user_id}
- POST /admin/api-keys/{user_id}/regenerate

Security model:
- The HTML is publicly accessible by default. The admin token is required to
  invoke admin APIs. Operators should protect access at the network layer
  (e.g., Cloudflare Access, VPN, IP allow-list) when exposing publicly.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/admin/ui", response_class=HTMLResponse)
async def admin_ui() -> str:
    """Serve a self-contained Admin UI for API key management.

    Returns:
      A small HTML document containing styles and JavaScript that interacts
      with the admin API endpoints using a user-provided admin token.
    """
    # Keep the HTML as a compact, dependency-free page. The UI stores the
    # token in memory (and optionally sessionStorage) and attaches it to each
    # admin API request.
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>HybridInference Admin UI</title>
    <style>
      :root {
        --bg: #0b1021;
        --panel: #121833;
        --text: #e5e7eb;
        --muted: #9aa3b2;
        --primary: #4f46e5;
        --danger: #ef4444;
        --ok: #10b981;
        --warn: #f59e0b;
        --border: #23304d;
      }
      body {
        margin: 0; font-family: ui-sans-serif, system-ui, -apple-system,
        Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue,
        Arial, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol;
        background: var(--bg); color: var(--text);
      }
      header { padding: 16px 20px; background: #0f1530; border-bottom: 1px solid var(--border); }
      header h1 { margin: 0; font-size: 18px; }
      main { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
      .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-bottom: 18px; }
      .row { display: flex; gap: 12px; flex-wrap: wrap; }
      .field { display: flex; flex-direction: column; gap: 6px; }
      label { color: var(--muted); font-size: 12px; }
      input, select, textarea {
        background: #0d132b; color: var(--text);
        border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
        font-size: 14px;
      }
      textarea { min-height: 60px; resize: vertical; }
      button { background: var(--primary); color: white; border: none; border-radius: 8px; padding: 8px 12px; cursor: pointer; font-weight: 600; }
      button.secondary { background: #2b3560; }
      button.danger { background: var(--danger); }
      button.warn { background: var(--warn); color: #1f2937; }
      button:disabled { opacity: 0.6; cursor: not-allowed; }
      .hint { color: var(--muted); font-size: 12px; }
      .tag { display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 12px; margin-right: 6px; border: 1px solid var(--border); background: #0f1530; }
      .tag.ok { border-color: #14532d; background: #083a23; color: #a7f3d0; }
      .tag.warn { border-color: #7c2d12; background: #3d1206; color: #fed7aa; }
      .tag.neutral { color: var(--muted); }
      table { width: 100%; border-collapse: collapse; }
      th, td { border-bottom: 1px solid var(--border); padding: 8px; text-align: left; font-size: 14px; }
      th { color: var(--muted); font-weight: 600; }
      .actions { display: flex; gap: 8px; flex-wrap: wrap; }
      .grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }
      @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .kbd { padding: 2px 6px; border: 1px solid var(--border); border-bottom-width: 2px; border-radius: 6px; background: #0f1530; font-size: 12px; }
      .note { background: #1a203f; border-left: 3px solid var(--warn); padding: 10px; border-radius: 6px; margin-top: 8px; }
      .success { background: #082c1f; border-left: 3px solid var(--ok); padding: 10px; border-radius: 6px; margin-top: 8px; }
      .error { background: #3b0d0d; border-left: 3px solid var(--danger); padding: 10px; border-radius: 6px; margin-top: 8px; }
      .copy { background: #2b3560; }
    </style>
  </head>
  <body>
    <header>
      <h1>HybridInference Admin UI</h1>
    </header>
    <main>
      <section class="card">
        <div class="row">
          <div class="field" style="min-width:280px;flex:1">
            <label for="token">Admin Token (Authorization Bearer)</label>
            <input id="token" type="password" placeholder="Paste ADMIN_TOKEN..." />
            <div class="hint">Token is kept in-memory; optionally remember in sessionStorage.</div>
          </div>
          <div class="field" style="align-self:flex-end;gap:10px;">
            <div class="row" style="gap:8px;">
              <button id="btn-set">Set Token</button>
              <button id="btn-clear" class="secondary">Clear</button>
              <label class="hint" style="display:flex;align-items:center;gap:6px;">
                <input type="checkbox" id="remember" /> Remember in session
              </label>
            </div>
            <div id="token-status" class="hint">No token set.</div>
          </div>
        </div>
      </section>

      <section class="card">
        <h3 style="margin-top:0">Create API Key</h3>
        <div class="grid">
          <div class="field">
            <label>User ID</label>
            <input id="user_id" placeholder="e.g., acme.alice" />
          </div>
          <div class="field">
            <label>User Name</label>
            <input id="user_name" placeholder="Optional" />
          </div>
          <div class="field">
            <label>Tier</label>
            <select id="tier">
              <option value="free">free</option>
              <option value="pro">pro</option>
              <option value="enterprise">enterprise</option>
            </select>
          </div>
          <div class="field">
            <label>Daily Cost Quota (USD)</label>
            <input id="quota_daily" type="number" step="0.01" value="1000" />
          </div>
          <div class="field">
            <label>Monthly Cost Quota (USD, optional)</label>
            <input id="quota_monthly" type="number" step="0.01" placeholder="" />
          </div>
          <div class="field">
            <label>Expires At (ISO 8601, optional)</label>
            <input id="expires_at" placeholder="2025-12-31T23:59:59Z" />
          </div>
        </div>
        <div class="field">
          <label>Notes</label>
          <textarea id="notes" placeholder="Admin notes (optional)"></textarea>
        </div>
        <div class="row" style="justify-content:flex-end; margin-top:8px;">
          <button id="btn-create">Create Key</button>
        </div>
        <div id="create-result" class="note" style="display:none"></div>
      </section>

      <section class="card">
        <div class="row" style="justify-content:space-between; align-items:center;">
          <h3 style="margin:0">API Keys</h3>
          <div class="row">
            <select id="filter-status">
              <option value="">All Status</option>
              <option value="active">active</option>
              <option value="suspended">suspended</option>
              <option value="revoked">revoked</option>
            </select>
            <select id="filter-tier">
              <option value="">All Tiers</option>
              <option value="free">free</option>
              <option value="pro">pro</option>
              <option value="enterprise">enterprise</option>
            </select>
            <button id="btn-refresh" class="secondary">Refresh</button>
          </div>
        </div>
        <div style="overflow:auto;">
          <table id="keys-table">
            <thead>
              <tr>
                <th>User</th>
                <th>Prefix</th>
                <th>Tier</th>
                <th>Status</th>
                <th>Usage (today / month)</th>
                <th>Created</th>
                <th>Expires</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
        <div id="list-error" class="error" style="display:none"></div>
      </section>

      <section class="card">
        <h3 style="margin-top:0">Key Detail</h3>
        <div id="detail" class="note">Select a key to view details.</div>
      </section>
    </main>

    <script>
      // Admin token handling
      let ADMIN_TOKEN = "";
      const tokenInput = document.getElementById('token');
      const tokenStatus = document.getElementById('token-status');
      const remember = document.getElementById('remember');
      let storageAvailable = true;

      function disableRememberOption() {
        remember.checked = false;
        remember.disabled = true;
        const rememberContainer = remember.parentElement;
        if (rememberContainer) {
          rememberContainer.classList.add('hint');
          rememberContainer.setAttribute('title', 'Session storage unavailable in this browser context.');
        }
      }

      function removeSessionToken() {
        if (!storageAvailable) return;
        try {
          sessionStorage.removeItem('ADMIN_TOKEN');
        } catch (err) {
          storageAvailable = false;
          console.warn('Failed to clear ADMIN_TOKEN from sessionStorage:', err);
          disableRememberOption();
        }
      }

      function storeSessionToken(token) {
        if (!storageAvailable) return;
        try {
          sessionStorage.setItem('ADMIN_TOKEN', token);
        } catch (err) {
          storageAvailable = false;
          console.warn('Failed to persist ADMIN_TOKEN to sessionStorage:', err);
          disableRememberOption();
        }
      }

      try {
        const savedToken = sessionStorage.getItem('ADMIN_TOKEN');
        if (savedToken) {
          ADMIN_TOKEN = savedToken;
          tokenStatus.textContent = 'Token set (session)';
          remember.checked = true;
        }
      } catch (err) {
        storageAvailable = false;
        console.warn('SessionStorage unavailable; disabling remember option:', err);
        disableRememberOption();
      }

      document.getElementById('btn-set').addEventListener('click', () => {
        ADMIN_TOKEN = tokenInput.value.trim();
        if (!remember.checked) removeSessionToken();
        if (!ADMIN_TOKEN) {
          tokenStatus.textContent = 'No token set.';
          return;
        }
        if (remember.checked && storageAvailable) {
          storeSessionToken(ADMIN_TOKEN);
          if (!storageAvailable) {
            disableRememberOption();
            tokenStatus.textContent = 'Token set';
            return;
          }
          tokenStatus.textContent = 'Token set (session)';
        } else {
          tokenStatus.textContent = 'Token set';
        }
      });

      document.getElementById('btn-clear').addEventListener('click', () => {
        ADMIN_TOKEN = "";
        tokenInput.value = "";
        removeSessionToken();
        tokenStatus.textContent = 'No token set.';
      });

      // Helpers
      function htmlesc(s) {
        return s == null ? '' : String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c] || c));
      }
      function fmtDate(s) {
        if (!s) return '';
        try {
          const date = new Date(s);
          if (Number.isNaN(date.getTime())) return String(s || '');
          return date.toISOString().replace('T', ' ').slice(0, -1) + 'Z';
        } catch {
          return String(s || '');
        }
      }
      function badge(text, cls) { return `<span class="tag ${cls||'neutral'}">${htmlesc(text)}</span>`; }
      function dollars(x){ if (x==null) return ''; const n = Number(x); if (Number.isNaN(n)) return String(x); return '$' + n.toFixed(2); }

      async function api(path, options={}) {
        if (!ADMIN_TOKEN) throw new Error('Admin token not set');
        const resp = await fetch(path, {
          headers: { 'Authorization': `Bearer ${ADMIN_TOKEN}`, 'Content-Type': 'application/json' },
          ...options,
        });
        const text = await resp.text();
        let json; try { json = text ? JSON.parse(text) : {}; } catch { json = { raw: text }; }
        if (!resp.ok) { const msg = json?.detail || json?.error || resp.statusText; throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg)); }
        return json;
      }

      // Create key
      document.getElementById('btn-create').addEventListener('click', async () => {
        const body = {
          user_id: document.getElementById('user_id').value.trim(),
          user_name: document.getElementById('user_name').value.trim() || null,
          tier: document.getElementById('tier').value,
          quota_daily_cost_usd: Number(document.getElementById('quota_daily').value || 1000),
          quota_monthly_cost_usd: document.getElementById('quota_monthly').value ? Number(document.getElementById('quota_monthly').value) : null,
          expires_at: document.getElementById('expires_at').value.trim() || null,
          notes: document.getElementById('notes').value.trim() || null,
          metadata: null,
        };
        const box = document.getElementById('create-result');
        box.style.display = 'block';
        box.className = 'note';
        box.textContent = 'Creating...';
        try {
          const res = await api('/admin/api-keys', { method: 'POST', body: JSON.stringify(body) });
          const key = htmlesc(res.api_key || '');
          box.className = 'success';
          box.innerHTML = `<div><strong>API key created.</strong> Save it now; it cannot be retrieved later.</div>
            <div style="margin-top:6px;" class="mono">${key}</div>`;
          refresh();
        } catch (e) {
          box.className = 'error';
          box.textContent = 'Error: ' + (e?.message || e);
        }
      });

      // List keys
      document.getElementById('btn-refresh').addEventListener('click', refresh);
      async function refresh() {
        const status = document.getElementById('filter-status').value || null;
        const tier = document.getElementById('filter-tier').value || null;
        const params = new URLSearchParams();
        if (status) params.set('status', status);
        if (tier) params.set('tier', tier);
        params.set('limit', '100');
        const url = '/admin/api-keys' + (params.toString() ? ('?' + params.toString()) : '');
        const err = document.getElementById('list-error');
        err.style.display = 'none';
        try {
          const data = await api(url);
          const tbody = document.querySelector('#keys-table tbody');
          tbody.innerHTML = '';
          for (const k of data.keys || []) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td><div><strong>${htmlesc(k.user_id)}</strong></div><div class="hint">${htmlesc(k.user_name||'')}</div></td>
              <td class="mono">${htmlesc(k.key_prefix)}</td>
              <td>${htmlesc(k.tier)}</td>
              <td>${k.status === 'active' ? badge('active','ok') : (k.status === 'revoked' ? badge('revoked','warn') : badge(k.status))}</td>
              <td>${dollars(k.usage_today_usd)} / ${dollars(k.usage_month_usd)}</td>
              <td>${fmtDate(k.created_at)}</td>
              <td>${fmtDate(k.expires_at)}</td>
              <td class="actions">
                <button class="secondary" data-action="detail" data-user="${htmlesc(k.user_id)}">Detail</button>
                <button class="warn" data-action="regen" data-user="${htmlesc(k.user_id)}">Regenerate</button>
                <button class="danger" data-action="revoke" data-user="${htmlesc(k.user_id)}">Revoke</button>
              </td>`;
            tbody.appendChild(tr);
          }
        } catch (e) {
          err.style.display = 'block';
          err.textContent = 'Error: ' + (e?.message || e);
        }
      }

      // Row actions
      document.querySelector('#keys-table tbody').addEventListener('click', async (ev) => {
        const btn = ev.target.closest('button'); if (!btn) return;
        const action = btn.getAttribute('data-action');
        const user = btn.getAttribute('data-user');
        if (!action || !user) return;
        if (action === 'detail') {
          try {
            const d = await api(`/admin/api-keys/${encodeURIComponent(user)}`);
            document.getElementById('detail').className = 'note';
            const usageToday = d.usage?.today || {}; const usageMonth = d.usage?.this_month || {};
            const models = (d.usage?.models_used||[]).map(m => `<span class="tag">${htmlesc(m)}</span>`).join(' ');
            document.getElementById('detail').innerHTML = `
              <div><strong>${htmlesc(d.user_id)}</strong> <span class="hint">${htmlesc(d.user_name||'')}</span></div>
              <div class="hint">Tier: ${htmlesc(d.tier)} | Status: ${htmlesc(d.status)} | Prefix: <span class="mono">${htmlesc(d.key_prefix)}</span></div>
              <div style="margin-top:8px;">
                <div>Today: ${dollars(usageToday.cost_usd)} (${htmlesc(String(usageToday.requests||0))} req)</div>
                <div>This Month: ${dollars(usageMonth.cost_usd)} (${htmlesc(String(usageMonth.requests||0))} req)</div>
              </div>
              <div style="margin-top:8px;">Models: ${models || '<span class="hint">No data</span>'}</div>
              <div class="hint" style="margin-top:8px;">Created: ${fmtDate(d.created_at)} | Last Used: ${fmtDate(d.last_used_at)} | Expires: ${fmtDate(d.expires_at)}</div>
              <div class="hint" style="margin-top:8px;">Notes: ${htmlesc(d.notes||'')}</div>
            `;
          } catch (e) {
            const box = document.getElementById('detail');
            box.className = 'error';
            box.textContent = 'Error: ' + (e?.message || e);
          }
        } else if (action === 'revoke') {
          if (!confirm(`Revoke key for ${user}?`)) return;
          try { await api(`/admin/api-keys/${encodeURIComponent(user)}`, { method: 'DELETE' }); refresh(); } catch (e) { alert('Error: ' + (e?.message || e)); }
        } else if (action === 'regen') {
          if (!confirm(`Regenerate key for ${user}? Old key will be invalidated.`)) return;
          try {
            const res = await api(`/admin/api-keys/${encodeURIComponent(user)}/regenerate`, { method: 'POST' });
            const key = res.api_key ? String(res.api_key) : '';
            const msg = `New key for ${user}:\n\n${key}\n\nSave it now; it cannot be retrieved later.`;
            alert(msg);
            refresh();
          } catch (e) { alert('Error: ' + (e?.message || e)); }
        }
      });

      // Initial paint
      // Do not auto-refresh to avoid 401 spam before token is set.
    </script>
  </body>
</html>
    """
